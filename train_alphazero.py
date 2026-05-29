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

    num_games = 50
    max_steps_per_game = 400

    learning_rate = 0.001

    num_simulations = 50
    c_puct = 1.5
    root_dirichlet_alpha = 0.3
    root_dirichlet_epsilon = 0.25

    # Temperature controls exploration when sampling from MCTS visit counts.
    # e.g. 1.0 = more exploration, 0.0 = always pick the most visited move.
    temperature = 1.0
    temperature_drop_step = 10

    use_training_ui = False
    ui_show_every = 1
    ui_speed = 30

    use_live_plotter = True

    # ============================================================
    # 2. PRINT THE CONFIGURATION
    # ============================================================
    print("\n=== AlphaZero Training Configuration ===")
    print(f"Games: {num_games}")
    print(f"Max steps per game: {max_steps_per_game}")
    print(f"Learning rate: {learning_rate}")
    print(f"MCTS simulations per move: {num_simulations}")
    print(f"c_puct: {c_puct}")
    print(f"Dirichlet alpha: {root_dirichlet_alpha}")
    print(f"Dirichlet epsilon: {root_dirichlet_epsilon}")
    print(f"Temperature: {temperature}")
    print(f"Temperature drop step: {temperature_drop_step}")
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
    env = GridGameAi()

    # Agent:
    # owns the policy-value network, AlphaZero trainer, and replay/examples buffer.
    agent = AlphaZeroSelfPlayAgent(
        lr=learning_rate,
        temperature=temperature,
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
        c_puct=c_puct,
        root_dirichlet_alpha=root_dirichlet_alpha,
        root_dirichlet_epsilon=root_dirichlet_epsilon,
        temperature=temperature,
        temperature_drop_step=temperature_drop_step,
    )


if __name__ == "__main__":
    main()
