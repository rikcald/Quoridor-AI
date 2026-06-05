#!/usr/bin/env python3

"""
Simple entrypoint for the AlphaZero-style training pipeline.

This file is intentionally written in a very direct style:
- hyperparameters are grouped in one visible block
- the whole pipeline is assembled inside main()
- there are no config classes or build helpers to chase around

Run from the project root with:
    venv\\Scripts\\python.exe train_alphazero.py
"""

from pathlib import Path
import sys


# Make the `game/` folder importable when this script is run from the repo root.
PROJECT_ROOT = Path(__file__).resolve().parent
GAME_DIR = PROJECT_ROOT / "game"
if str(GAME_DIR) not in sys.path:
    sys.path.insert(0, str(GAME_DIR))

import alphazero_training as training_module
from agent import AlphaZeroSelfPlayAgent
from alphazero_training import train_alphazero_self_play
from game_logic_Ai import GridGameAi
from helper import LivePlotter
from model import DEVICE
from pygame_training_ui import TrainingUI


def main():
    # ============================================================
    # 1. TRAINING SETTINGS
    # ============================================================

    # Board variant:
    # - fast exam-friendly training: board_size=5, max_walls=4
    # - classic Quoridor: board_size=9, max_walls=10
    board_size = 5
    max_walls = 4

    num_games = 50
    max_steps_per_game = 100

    learning_rate = 0.001

    num_simulations = 50
    mcts_batch_size = 1
    c_puct = 1.5
    root_dirichlet_alpha = 0.3
    root_dirichlet_epsilon = 0.25

    # Action temperature controls how self-play chooses the move to actually play.
    # e.g. 1.0 samples from visit counts; 0.0 always picks the most visited move.
    temperature = 1.0
    temperature_after_drop = 0.0
    temperature_drop_step = 10
    # Target policy temperature controls what the network learns from MCTS visits.
    # e.g. keep this at 1.0 to train on a soft 70/20/10 target, even if action
    # temperature drops to 0.0 and self-play chooses greedily.
    target_policy_temperature = 1.0
    timeout_adjudication_value = 0.0  # If a game hits the max step limit, assign this value to both players for training purposes.

    use_training_ui = False
    ui_show_every = 1
    ui_speed = 30

    use_live_plotter = True

    # ============================================================
    # 2. PRINT THE CONFIGURATION
    # ============================================================
    print("\n=== AlphaZero Training Configuration ===")
    print(f"Board size: {board_size}x{board_size}")
    print(f"Max walls per player: {max_walls}")
    print(f"Games: {num_games}")
    print(f"Max steps per game: {max_steps_per_game}")
    print(f"Learning rate: {learning_rate}")
    print(f"MCTS simulations per move: {num_simulations}")
    print(f"MCTS inference batch size: {mcts_batch_size}")
    print(f"c_puct: {c_puct}")
    print(f"Dirichlet alpha: {root_dirichlet_alpha}")
    print(f"Dirichlet epsilon: {root_dirichlet_epsilon}")
    print(f"Action temperature: {temperature}")
    print(f"Action temperature after drop: {temperature_after_drop}")
    print(f"Temperature drop step: {temperature_drop_step}")
    print(f"Target policy temperature: {target_policy_temperature}")
    print(f"Timeout adjudication value: {timeout_adjudication_value}")
    print(f"Use training UI: {use_training_ui}")
    print(f"UI show every: {ui_show_every}")
    print(f"UI speed: {ui_speed}")
    print(f"Use live plotter: {use_live_plotter}")
    print(f"PyTorch device: {DEVICE}")
    print("========================================\n")

    # ============================================================
    # 3. CREATE THE MAIN OBJECTS
    # ============================================================
    # Environment:
    # owns the board state, legal moves, winner detection, canonical state, etc.
    env = GridGameAi(grid_size=board_size, max_walls=max_walls)
    print(f"Total action slots: {env.total_actions}")

    # Agent:
    # owns the policy-value network, AlphaZero trainer, and replay/examples buffer.
    agent = AlphaZeroSelfPlayAgent(
        lr=learning_rate,
        temperature=temperature,
        board_size=board_size,
        max_walls=max_walls,
        num_actions=env.total_actions,
    )

    # Optional pygame visualization.
    if use_training_ui:
        ui = TrainingUI(
            env=env,
            show_every=ui_show_every,
            speed=ui_speed,
        )
    else:
        ui = None

    # Optional live matplotlib plotter.
    if use_live_plotter:
        plotter = LivePlotter()
    else:
        plotter = None

    # ============================================================
    # 4. OVERRIDE TRAINING-LOOP GLOBAL SETTINGS
    # ============================================================
    # The current training loop reads MAX_STEPS_PER_GAME from game/alphazero_training.py.
    # We set it here so this script is the obvious place to control it.
    training_module.MAX_STEPS_PER_GAME = max_steps_per_game

    # ============================================================
    # 5. START TRAINING
    # ============================================================
    # The loop below will:
    # - run MCTS at each move
    # - turn root visit counts into a target policy pi
    # - store (state, pi, z) examples
    # - train the policy-value network from those examples
    train_alphazero_self_play(
        env=env,
        agent=agent,
        plotter=plotter,
        ui=ui,
        num_games=num_games,
        num_simulations=num_simulations,
        mcts_batch_size=mcts_batch_size,
        c_puct=c_puct,
        root_dirichlet_alpha=root_dirichlet_alpha,
        root_dirichlet_epsilon=root_dirichlet_epsilon,
        temperature=temperature,
        temperature_after_drop=temperature_after_drop,
        temperature_drop_step=temperature_drop_step,
        target_policy_temperature=target_policy_temperature,
        timeout_adjudication_value=timeout_adjudication_value,
    )


if __name__ == "__main__":
    main()
