from pathlib import Path
import csv
import json
from datetime import datetime

import numpy as np

from game_logic_Ai import P1, P2, NUM_MOVE_ACTIONS
from mcts import MCTS


SCRIPT_DIR = Path(__file__).resolve().parent
PLOTS_DIR = SCRIPT_DIR.parent / "plots"
PLOTS_DIR.mkdir(exist_ok=True)
METRICS_DIR = SCRIPT_DIR.parent / "training_metrics"
METRICS_DIR.mkdir(exist_ok=True)
RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

MAX_STEPS_PER_GAME = 400
ALPHAZERO_BATCH_SIZE = 64
DEFAULT_TIMEOUT_ADJUDICATION_VALUE = 0


def train_alphazero_self_play(
    env,
    agent,
    plotter,
    ui=None,
    num_games=100,
    num_simulations=50,
    c_puct=1.5,
    root_dirichlet_alpha=0.3,
    root_dirichlet_epsilon=0.25,
    temperature=1.0,
    temperature_drop_step=10,
    mcts_batch_size=1,
    timeout_adjudication_value=DEFAULT_TIMEOUT_ADJUDICATION_VALUE,
):
    """
    Full AlphaZero-style self-play loop.

    Per move:
    1. run MCTS from the current position
    2. convert root visit counts into target policy pi
    3. sample an action from pi
    4. store (state, pi, player)

    At the end of the game:
    5. assign final z values
    6. train the policy-value network on the collected examples
    """
    record_value_loss = float("inf")
    stats = {
        "steps": [],
        "p1_rewards": [],
        "p2_rewards": [],
        "walls": [],
        "p1_wins": [],
        "p2_wins": [],
        "exploration_rate_p1": [],
        "exploration_rate_p2": [],
        "invalid_moves": [],
        "shared_scores": [],
        "timeouts": [],
        "adjudicated_timeouts": [],
        "policy_loss": [],
        "value_loss": [],
        "total_loss": [],
        "examples_per_game": [],
    }

    metrics_csv_path = METRICS_DIR / f"alphazero_self_play_metrics_{RUN_TIMESTAMP}.csv"
    metrics_json_path = (
        METRICS_DIR / f"alphazero_self_play_summary_{RUN_TIMESTAMP}.json"
    )

    for game_num in range(num_games):
        env.reset()
        agent.start_new_game()
        done = False
        step_count = 0
        wall_count = 0
        invalid_count = 0
        root_temperature = temperature

        print(
            f"\n================ Starting AlphaZero Self-Play Game {game_num + 1} ================"
        )

        if ui is not None:
            ui.new_game()

        while not done and step_count < MAX_STEPS_PER_GAME:
            current_player = env.get_current_player()
            canonical_state = env.get_canonical_state(current_player)

            search = MCTS(
                agent=agent,
                num_simulations=num_simulations,
                c_puct=c_puct,
                dirichlet_alpha=root_dirichlet_alpha,
                dirichlet_epsilon=root_dirichlet_epsilon,
                add_dirichlet_noise=True,
            )
            if mcts_batch_size > 1:
                root = search.run_batched(env, batch_size=mcts_batch_size)
            else:
                root = search.run(env)

            # Early in the game we keep more exploration, later we collapse
            # toward the most visited move.
            root_temperature = (
                temperature if step_count < temperature_drop_step else 0.0
            )
            target_policy = search.get_action_probs(root, temperature=root_temperature)

            agent.record_policy_example(
                state=canonical_state,
                target_policy=target_policy,
                player=current_player,
            )

            action = search.select_action(root, temperature=root_temperature)
            _, done, info = env.apply_action(action)

            if info.get("invalid", False):
                invalid_count += 1
            elif action >= NUM_MOVE_ACTIONS:
                wall_count += 1

            step_count += 1

            if ui is not None:
                ui.render(game_num=game_num + 1, step=step_count)

        timeout_reached = step_count >= MAX_STEPS_PER_GAME and not env.is_terminal()
        adjudicated_timeout = False
        target_value_strength = 1.0

        if timeout_reached:
            # A max-step cutoff is not a real terminal state. Instead of
            # teaching the value head that every unfinished game is neutral,
            # give a weak +/- signal to the player with the shorter legal path.
            winner = env.get_timeout_adjudication_winner()
            target_value_strength = timeout_adjudication_value
            adjudicated_timeout = winner is not None
        else:
            winner = env.get_winner()

        finalized_examples = agent.finalize_game_examples(
            winner=winner,
            outcome_value=target_value_strength,
        )
        train_info = agent.train_from_examples(batch_size=ALPHAZERO_BATCH_SIZE)

        p1_outcome = (
            0.0
            if winner is None
            else (target_value_strength if winner == P1 else -target_value_strength)
        )
        p2_outcome = -p1_outcome
        shared_score = p1_outcome + p2_outcome

        if train_info is None:
            train_info = {
                "total_loss": 0.0,
                "policy_loss": 0.0,
                "value_loss": 0.0,
            }

        if train_info["value_loss"] < record_value_loss:
            record_value_loss = train_info["value_loss"]
            agent.save_checkpoint(
                "PolicyValueNet_alphazero_best_value_loss.pth",
                extra_metadata={
                    "game_num": game_num + 1,
                    "value_loss": record_value_loss,
                    "policy_loss": train_info["policy_loss"],
                    "total_loss": train_info["total_loss"],
                },
            )

        agent.save_checkpoint(
            "PolicyValueNet_alphazero_latest.pth",
            extra_metadata={
                "game_num": game_num + 1,
                "value_loss": train_info["value_loss"],
                "policy_loss": train_info["policy_loss"],
                "total_loss": train_info["total_loss"],
                "examples_in_buffer": len(agent.examples),
            },
        )

        stats["steps"].append(step_count)
        stats["p1_rewards"].append(p1_outcome)
        stats["p2_rewards"].append(p2_outcome)
        stats["walls"].append(wall_count)
        stats["p1_wins"].append(1 if not timeout_reached and winner == P1 else 0)
        stats["p2_wins"].append(1 if not timeout_reached and winner == P2 else 0)
        stats["invalid_moves"].append(invalid_count)
        stats["shared_scores"].append(shared_score)
        stats["timeouts"].append(1 if timeout_reached else 0)
        stats["adjudicated_timeouts"].append(1 if adjudicated_timeout else 0)
        # Reuse the old plot layout: here "exploration" means root temperature.
        stats["exploration_rate_p1"].append(root_temperature)
        stats["exploration_rate_p2"].append(root_temperature)
        stats["policy_loss"].append(train_info["policy_loss"])
        stats["value_loss"].append(train_info["value_loss"])
        stats["total_loss"].append(train_info["total_loss"])
        stats["examples_per_game"].append(len(finalized_examples))
        if plotter is not None:
            plotter.update(stats)

        _append_metrics_row(
            metrics_csv_path,
            {
                "game": game_num + 1,
                "steps": step_count,
                "winner": winner,
                "timeout": int(timeout_reached),
                "adjudicated_timeout": int(adjudicated_timeout),
                "target_value_strength": target_value_strength,
                "walls": wall_count,
                "invalid_moves": invalid_count,
                "examples_generated": len(finalized_examples),
                "replay_size": len(agent.examples),
                "policy_loss": round(train_info["policy_loss"], 6),
                "value_loss": round(train_info["value_loss"], 6),
                "total_loss": round(train_info["total_loss"], 6),
                "root_temperature": root_temperature,
                "num_simulations": num_simulations,
                "mcts_batch_size": mcts_batch_size,
            },
        )

        _write_alphazero_metrics_summary(
            metrics_json_path,
            stats,
            latest_game=game_num + 1,
            latest_winner=winner,
            replay_size=len(agent.examples),
            num_simulations=num_simulations,
            mcts_batch_size=mcts_batch_size,
        )

        print(
            f"\nGame {game_num + 1} | "
            f"Winner: {winner} | "
            f"Steps: {step_count} | "
            f"Timeout adjudicated: {adjudicated_timeout} | "
            f"Examples: {len(finalized_examples)} | "
            f"Replay: {len(agent.examples)} | "
            f"Policy Loss: {train_info['policy_loss']:.4f} | "
            f"Value Loss: {train_info['value_loss']:.4f} | "
            f"Total Loss: {train_info['total_loss']:.4f}"
        )

    if plotter is not None:
        plotter.fig.savefig(
            str(PLOTS_DIR / "training_alphazero_self_play.png"),
            dpi=300,
            bbox_inches="tight",
        )


def _append_metrics_row(csv_path, row):
    fieldnames = list(row.keys())
    write_header = not csv_path.exists()

    with csv_path.open("a", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _write_alphazero_metrics_summary(
    metrics_json_path,
    stats,
    latest_game,
    latest_winner,
    replay_size,
    num_simulations,
    mcts_batch_size,
):
    summary = {
        "latest_game": latest_game,
        "latest_winner": latest_winner,
        "replay_size": replay_size,
        "num_simulations": num_simulations,
        "mcts_batch_size": mcts_batch_size,
        "total_p1_wins": int(sum(stats["p1_wins"])),
        "total_p2_wins": int(sum(stats["p2_wins"])),
        "timeouts": int(sum(stats["timeouts"])),
        "adjudicated_timeouts": int(sum(stats["adjudicated_timeouts"])),
        "avg_steps": float(np.mean(stats["steps"])) if stats["steps"] else 0.0,
        "avg_policy_loss": (
            float(np.mean(stats["policy_loss"])) if stats["policy_loss"] else 0.0
        ),
        "avg_value_loss": (
            float(np.mean(stats["value_loss"])) if stats["value_loss"] else 0.0
        ),
        "avg_total_loss": (
            float(np.mean(stats["total_loss"])) if stats["total_loss"] else 0.0
        ),
        "avg_examples_per_game": (
            float(np.mean(stats["examples_per_game"]))
            if stats["examples_per_game"]
            else 0.0
        ),
    }

    with metrics_json_path.open("w", encoding="utf-8") as metrics_file:
        json.dump(summary, metrics_file, indent=2)
