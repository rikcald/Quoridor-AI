from collections import deque
from pathlib import Path
import csv
import json
from datetime import datetime

import wandb
import torch
import random
import numpy as np
from pygame_training_ui import TrainingUI
from game_logic_Ai import P1, P2, TOTAL_ACTIONS, NUM_MOVE_ACTIONS, GridGameAi
from model import Linear_QNet, QTrainer, PolicyValueNet, AlphaZeroTrainer
from mcts import MCTS
from helper import LivePlotter

SCRIPT_DIR = Path(__file__).resolve().parent
PLOTS_DIR = SCRIPT_DIR.parent / "plots"
PLOTS_DIR.mkdir(exist_ok=True)
MODEL_DIR = SCRIPT_DIR.parent / "model"
MODEL_DIR.mkdir(exist_ok=True)
METRICS_DIR = SCRIPT_DIR.parent / "training_metrics"
METRICS_DIR.mkdir(exist_ok=True)
RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

MAX_MEMORY = 100000
BATCH_SIZE = 1000

NUM_GAMES = 500
MAX_STEPS_PER_GAME = 400
RANDOM_MOVE_WALL_PROB = 0.5

LR = 0.001
TIMEOUT_PENALTY = -5.0
ALPHAZERO_BATCH_SIZE = 64


class AlphaZeroExample:
    """
    One supervised learning example produced by self-play.

    Fields:
    - state: canonical board tensor from the acting player's perspective
    - target_policy: probability distribution over the 144 action slots
    - player: the real player id that owned this state during the game
    - target_value: final game result from that player's perspective, filled in later

    - temperature controlla quanto l'agente esplora invece di scegliere sempre la mossa più probabile. 1.0 -> più esplorazione, 0.0 -> sempre la mossa più probabile
    """

    def __init__(self, state, target_policy, player, target_value=None):
        self.state = state
        self.target_policy = target_policy
        self.player = player
        self.target_value = target_value


class SelfPlayAgent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0
        self.epsilon_decay = 4
        self.gamma = 0.9
        self.memory = deque(maxlen=MAX_MEMORY)
        # One shared network learns from both sides of the board.
        # e.g. the same weights act as P1 on one turn and P2 on the next.
        self.model = Linear_QNet(486, TOTAL_ACTIONS)
        # The trainer uses an alternating zero-sum bootstrap:
        # e.g. after my move, the next canonical state belongs to the opponent,
        # so its best value is subtracted instead of added.
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def save_checkpoint(self, file_name, extra_metadata=None):
        # Save richer training state than a plain model-only .pth file:
        # e.g. n_games and epsilon are useful if later you want to resume training.
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "n_games": self.n_games,
            "epsilon": self.epsilon,
            "gamma": self.gamma,
        }

        if extra_metadata is not None:
            checkpoint["metadata"] = extra_metadata

        torch.save(checkpoint, MODEL_DIR / file_name)

    def get_action(self, state, env, player):
        # Self-play keeps a single exploration schedule because the policy is shared.
        self.epsilon = max(5, 80 - self.n_games // self.epsilon_decay)

        if random.randint(0, 100) < self.epsilon:
            mask = env.get_action_mask(player)
            valid_actions = np.where(mask == 1)[0]

            if len(valid_actions) == 0:
                raise Exception(f"No valid actions available for player {player}")

            valid_moves = valid_actions[valid_actions < NUM_MOVE_ACTIONS]
            valid_walls = valid_actions[valid_actions >= NUM_MOVE_ACTIONS]

            # Mild bias toward movement during exploration:
            # e.g. early training should still discover progress toward the goal,
            # instead of wasting most turns on random wall placements.
            if random.random() < RANDOM_MOVE_WALL_PROB and len(valid_moves) > 0:
                return int(np.random.choice(valid_moves))
            elif len(valid_walls) > 0:
                return int(np.random.choice(valid_walls))

            return int(np.random.choice(valid_actions))

        # Illegal actions are masked before argmax:
        # e.g. a high-Q wall action is ignored if that wall cannot be placed now.
        state_tensor = torch.tensor(state, dtype=torch.float).flatten()
        prediction = self.model(state_tensor.unsqueeze(0))

        mask = env.get_action_mask(player)
        mask_tensor = torch.tensor(mask, dtype=torch.float)

        masked_prediction = prediction.clone()
        masked_prediction[0, mask_tensor == 0] = -1e9
        return torch.argmax(masked_prediction).item()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def train_long_memory(self):
        if not self.memory:
            return

        # Replay samples contain experiences generated by both players, but all of them
        # live in the same canonical perspective space used for self-play.
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)

        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        next_states = np.array(next_states)
        dones = np.array(dones)

        self.trainer.train_step(states, actions, rewards, next_states, dones)


class AlphaZeroSelfPlayAgent:
    """
    Separate AlphaZero-style agent kept alongside the old DQN pipeline.

    This class does not learn from (state, action, reward, next_state).
    Instead it learns from full self-play examples of the form:
    - state
    - target policy pi from MCTS visit counts
    - final value z from the game outcome
    """

    def __init__(self, lr=0.001, temperature=1.0):
        self.n_games = 0
        self.temperature = temperature
        self.model = PolicyValueNet(
            input_channels=6,
            board_size=9,
            num_actions=TOTAL_ACTIONS,
        )
        self.trainer = AlphaZeroTrainer(self.model, lr=lr)
        self.examples = deque(maxlen=MAX_MEMORY)
        self.current_game_examples = []

    def start_new_game(self):
        self.current_game_examples = []

    def predict(self, state):
        """
        Run the policy-value network on one canonical board state.

        Returns:
        - policy probabilities over the 144 shared action slots
        - scalar value estimate in [-1, 1]
        """
        return self.model.predict(state)

    def mask_and_normalize_policy(self, policy, valid_actions):
        """
        Zero-out illegal actions and renormalize the legal ones.

        e.g. if the raw network likes action 17 but action 17 is illegal here,
        that probability mass is removed before sampling a move.
        """
        masked_policy = np.zeros(TOTAL_ACTIONS, dtype=np.float32)
        masked_policy[valid_actions] = policy[valid_actions]

        total_prob = float(masked_policy.sum())
        if total_prob <= 0:
            # Fallback to uniform over valid actions if the raw network mass collapses.
            masked_policy[valid_actions] = 1.0 / len(valid_actions)
            return masked_policy

        masked_policy /= total_prob
        return masked_policy

    def select_action_from_policy(self, policy, valid_actions, temperature=None):
        """
        Sample one action from a masked policy distribution.

        Later, when MCTS is added, `policy` will usually be the normalized visit
        counts from the root rather than the raw network prediction.
        """
        if temperature is None:
            temperature = self.temperature

        masked_policy = self.mask_and_normalize_policy(policy, valid_actions)

        if temperature <= 1e-8:
            return int(np.argmax(masked_policy))

        # Apply temperature in probability space for a simple readable first version.
        tempered_policy = masked_policy ** (1.0 / temperature)
        tempered_policy /= tempered_policy.sum()
        return int(np.random.choice(np.arange(TOTAL_ACTIONS), p=tempered_policy))

    def record_policy_example(self, state, target_policy, player):
        """
        Store one unfinished AlphaZero example for the current game.

        `target_value` is not known yet and will be filled in only when the game ends.
        """
        self.current_game_examples.append(
            AlphaZeroExample(
                state=np.array(state, dtype=np.float32, copy=True),
                target_policy=np.array(target_policy, dtype=np.float32, copy=True),
                player=player,
                target_value=None,
            )
        )

    def finalize_game_examples(self, winner):
        """
        Convert unfinished per-move examples into trainable (state, pi, z) data.

        e.g. if a stored example belongs to P1 and P1 later wins, its z becomes +1.
        If the same finished game contains an example belonging to P2, its z becomes -1.
        """
        finalized_examples = []

        for example in self.current_game_examples:
            if winner is None:
                example.target_value = 0.0
            else:
                example.target_value = 1.0 if example.player == winner else -1.0

            finalized_examples.append(example)
            self.examples.append(example)

        self.current_game_examples = []
        self.n_games += 1
        return finalized_examples

    def train_from_examples(self, batch_size=ALPHAZERO_BATCH_SIZE):
        """
        Train the policy-value network on finalized self-play examples.

        This is the AlphaZero-style replacement for replaying Q-learning transitions.
        """
        if not self.examples:
            return None

        if len(self.examples) > batch_size:
            mini_batch = random.sample(self.examples, batch_size)
        else:
            mini_batch = list(self.examples)

        states = np.array([example.state for example in mini_batch], dtype=np.float32)
        target_policies = np.array(
            [example.target_policy for example in mini_batch], dtype=np.float32
        )
        target_values = np.array(
            [example.target_value for example in mini_batch], dtype=np.float32
        )

        return self.trainer.train_batch(states, target_policies, target_values)

    def save_checkpoint(self, file_name, extra_metadata=None):
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "n_games": self.n_games,
            "temperature": self.temperature,
        }

        if extra_metadata is not None:
            checkpoint["metadata"] = extra_metadata

        torch.save(checkpoint, MODEL_DIR / file_name)


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
):
    """
    Full AlphaZero-style self-play loop.

    Per move:
    1. run MCTS from the current position
    2. convert root visit counts into target policy pi
    3. sample an action from pi
    4. store (state, pi, player)

    At the end of the game:
    5. assign final z values to all stored positions
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
        "policy_loss": [],
        "value_loss": [],
        "total_loss": [],
        "examples_per_game": [],
    }

    metrics_csv_path = METRICS_DIR / f"alphazero_self_play_metrics_{RUN_TIMESTAMP}.csv"
    metrics_json_path = METRICS_DIR / f"alphazero_self_play_summary_{RUN_TIMESTAMP}.json"

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
            root = search.run(env)

            # Early in the game we keep more exploration, later we can collapse
            # toward the most visited move.
            root_temperature = temperature if step_count < temperature_drop_step else 0.0
            target_policy = search.get_action_probs(root, temperature=root_temperature)

            agent.record_policy_example(
                state=canonical_state,
                target_policy=target_policy,
                player=current_player,
            )

            action = search.select_action(root, temperature=root_temperature)
            _, _, done, info = env.apply_action(action)

            if info.get("invalid", False):
                invalid_count += 1
            elif action >= NUM_MOVE_ACTIONS:
                wall_count += 1

            step_count += 1

            if ui is not None:
                ui.render(game_num=game_num + 1, step=step_count)

        timeout_reached = step_count >= MAX_STEPS_PER_GAME and not env.is_terminal()
        winner = None if timeout_reached else env.get_winner()
        finalized_examples = agent.finalize_game_examples(winner=winner)
        train_info = agent.train_from_examples(batch_size=ALPHAZERO_BATCH_SIZE)

        p1_outcome = 0.0 if winner is None else (1.0 if winner == P1 else -1.0)
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
        stats["p1_wins"].append(1 if winner == P1 else 0)
        stats["p2_wins"].append(1 if winner == P2 else 0)
        stats["invalid_moves"].append(invalid_count)
        stats["shared_scores"].append(shared_score)
        stats["timeouts"].append(1 if timeout_reached else 0)
        # Reuse the plot layout: here "exploration" means root temperature, not epsilon.
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
                "walls": wall_count,
                "invalid_moves": invalid_count,
                "examples_generated": len(finalized_examples),
                "replay_size": len(agent.examples),
                "policy_loss": round(train_info["policy_loss"], 6),
                "value_loss": round(train_info["value_loss"], 6),
                "total_loss": round(train_info["total_loss"], 6),
                "root_temperature": root_temperature,
                "num_simulations": num_simulations,
            },
        )

        _write_alphazero_metrics_summary(
            metrics_json_path,
            stats,
            latest_game=game_num + 1,
            latest_winner=winner,
            replay_size=len(agent.examples),
        )

        print(
            f"\nGame {game_num + 1} | "
            f"Winner: {winner} | "
            f"Steps: {step_count} | "
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


def train_self_play(env, agent, plotter, ui=None, num_games=1000):
    record_score = float("-inf")
    best_win_balance = float("-inf")
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
    }

    metrics_csv_path = METRICS_DIR / f"self_play_metrics_{RUN_TIMESTAMP}.csv"
    metrics_json_path = METRICS_DIR / f"self_play_summary_{RUN_TIMESTAMP}.json"

    for game_num in range(num_games):
        # Each episode is symmetric self-play: turns alternate in the environment,
        # but both turns are controlled by the same network instance.
        env.reset()
        done = False
        step_count = 0
        p1_episode_reward = 0.0
        p2_episode_reward = 0.0

        print(
            f"\n================ Starting Self-Play Game {game_num + 1} ================"
        )

        move_count = 0
        wall_count = 0
        invalid_count = 0

        if ui is not None:
            ui.new_game()

        while not done and step_count < MAX_STEPS_PER_GAME:
            current_player = env.turn
            state_old = env.get_state()
            action = agent.get_action(state_old, env, current_player)

            _, reward, done, info = env.step(action)

            # The environment switches turn inside step(). For shared self-play we keep
            # the transition in the acting player's canonical frame:
            # e.g. if P1 just moved, next_state must still be encoded from P1's view.
            next_state_for_actor = env.get_state_player_centric(current_player)

            if info.get("invalid", False):
                invalid_count += 1
            elif action < NUM_MOVE_ACTIONS:
                move_count += 1
            else:
                wall_count += 1

            agent.train_short_memory(
                state_old, action, reward, next_state_for_actor, done
            )
            agent.remember(state_old, action, reward, next_state_for_actor, done)

            if current_player == P1:
                p1_episode_reward += reward
            else:
                p2_episode_reward += reward

            step_count += 1

            if ui is not None:
                ui.render(game_num=game_num + 1, step=step_count)

        timeout_reached = step_count >= MAX_STEPS_PER_GAME

        if timeout_reached:
            # Timeout means the shared policy failed to convert the game into a result,
            # so both sides receive the same penalty.
            p1_episode_reward += TIMEOUT_PENALTY
            p2_episode_reward += TIMEOUT_PENALTY

        agent.n_games += 1
        agent.train_long_memory()

        # Save the shared model when the combined game score improves.
        # e.g. this rewards faster wins and fewer invalid/stalled games on both sides.
        final_score = p1_episode_reward + p2_episode_reward
        if final_score > record_score:
            record_score = final_score
            agent.save_checkpoint(
                "Linear_QNet_self_play_best_score.pth",
                extra_metadata={
                    "record_score": record_score,
                    "game_num": game_num + 1,
                    "step_count": step_count,
                },
            )

        stats["steps"].append(step_count)
        stats["p1_rewards"].append(p1_episode_reward)
        stats["p2_rewards"].append(p2_episode_reward)
        stats["walls"].append(wall_count)
        stats["p1_wins"].append(1 if info.get("winner") == P1 else 0)
        stats["p2_wins"].append(1 if info.get("winner") == P2 else 0)
        stats["invalid_moves"].append(invalid_count)
        stats["shared_scores"].append(final_score)
        stats["timeouts"].append(1 if timeout_reached else 0)

        next_epsilon = max(5, 80 - agent.n_games // agent.epsilon_decay)
        # The model is shared, but we log the same epsilon on both curves to keep
        # the existing plotting layout stable.
        stats["exploration_rate_p1"].append(next_epsilon)
        stats["exploration_rate_p2"].append(next_epsilon)
        plotter.update(stats)

        winner = info.get("winner")
        win_balance = stats["p1_wins"][-1] - stats["p2_wins"][-1]
        if winner is not None and win_balance > best_win_balance:
            best_win_balance = win_balance

        # CSV is append-only so long runs remain inspectable even if training stops early.
        _append_metrics_row(
            metrics_csv_path,
            {
                "game": game_num + 1,
                "steps": step_count,
                "winner": winner,
                "p1_reward": round(p1_episode_reward, 4),
                "p2_reward": round(p2_episode_reward, 4),
                "shared_score": round(final_score, 4),
                "moves": move_count,
                "walls": wall_count,
                "invalid_moves": invalid_count,
                "timeout": int(timeout_reached),
                "epsilon": next_epsilon,
            },
        )

        agent.save_checkpoint(
            "Linear_QNet_self_play_latest.pth",
            extra_metadata={
                "game_num": game_num + 1,
                "record_score": record_score,
                "last_shared_score": final_score,
            },
        )

        _write_metrics_summary(
            metrics_json_path,
            stats,
            latest_game=game_num + 1,
            record_score=record_score,
            latest_winner=winner,
        )

        if agent.n_games % 2 == 0:
            print(
                f"\nGame {agent.n_games} | "
                f"P1 Score: {p1_episode_reward:.2f} | "
                f"P2 Score: {p2_episode_reward:.2f} | "
                f"Shared Score: {final_score:.2f} (Record: {record_score:.2f}) | "
                f"Steps: {step_count} | "
                f"Moves: {move_count}, Walls: {wall_count}, Invalid: {invalid_count} | "
                f"Epsilon: {next_epsilon}"
            )

    plotter.fig.savefig(
        str(PLOTS_DIR / "training_self_play.png"), dpi=300, bbox_inches="tight"
    )
    """run.log({"training_plot": wandb.Image(str(PLOTS_DIR / "training_self_play.png"))})"""


def _append_metrics_row(csv_path, row):
    fieldnames = list(row.keys())
    write_header = not csv_path.exists()

    # e.g. after 10,000 games you can load this CSV in pandas or Excel and inspect trends.
    with csv_path.open("a", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _write_metrics_summary(
    metrics_json_path, stats, latest_game, record_score, latest_winner
):
    # Lightweight summary for quick checks without parsing the full per-game CSV.
    summary = {
        "latest_game": latest_game,
        "record_score": record_score,
        "latest_winner": latest_winner,
        "total_p1_wins": int(sum(stats["p1_wins"])),
        "total_p2_wins": int(sum(stats["p2_wins"])),
        "timeouts": int(sum(stats["timeouts"])),
        "avg_steps": float(np.mean(stats["steps"])) if stats["steps"] else 0.0,
        "avg_shared_score": (
            float(np.mean(stats["shared_scores"])) if stats["shared_scores"] else 0.0
        ),
        "avg_invalid_moves": (
            float(np.mean(stats["invalid_moves"])) if stats["invalid_moves"] else 0.0
        ),
    }

    with metrics_json_path.open("w", encoding="utf-8") as metrics_file:
        json.dump(summary, metrics_file, indent=2)


def _write_alphazero_metrics_summary(
    metrics_json_path, stats, latest_game, latest_winner, replay_size
):
    summary = {
        "latest_game": latest_game,
        "latest_winner": latest_winner,
        "replay_size": replay_size,
        "total_p1_wins": int(sum(stats["p1_wins"])),
        "total_p2_wins": int(sum(stats["p2_wins"])),
        "timeouts": int(sum(stats["timeouts"])),
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


if __name__ == "__main__":
    """run = wandb.init(
        project="quoridor-rl",
        name="training_self_play_v0",
        config={
            "architecture": "Linear_QNet",
            "epoches": NUM_GAMES,
            "max_steps_per_game": MAX_STEPS_PER_GAME,
            "learning_rate": LR,
            "gamma": 0.9,
            "epsilon_start": 80,
            "epsilon_decay": 4,
            "epsilon_min": 5,
            "batch_size": BATCH_SIZE,
            "max_memory": MAX_MEMORY,
            "training_mode": "shared-model self-play",
        },
    )"""

    print("\n=== Starting Shared-Model Self-Play Training ===")
    agent = SelfPlayAgent()
    env = GridGameAi()
    ui = TrainingUI(env, show_every=2, speed=30)

    plotter = LivePlotter()
    train_self_play(env, agent, plotter, ui, num_games=NUM_GAMES)
    """run.finish()"""
