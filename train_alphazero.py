#!/usr/bin/env python3

"""
Dedicated entrypoint for the AlphaZero-style training pipeline.

Why this file exists:
- it keeps the AlphaZero workflow separate from the older DQN/Q-learning code
- it gives one simple command to launch training
- it collects the main hyperparameters in one easy-to-edit place

Run from the project root with:
    venv\\Scripts\\python.exe train_alphazero.py
"""

from dataclasses import dataclass
from pathlib import Path
import sys


# Make the `game/` folder importable when this script is run from the repo root.
PROJECT_ROOT = Path(__file__).resolve().parent
GAME_DIR = PROJECT_ROOT / "game"
if str(GAME_DIR) not in sys.path:
    sys.path.insert(0, str(GAME_DIR))

from agent import AlphaZeroSelfPlayAgent, train_alphazero_self_play
from game_logic_Ai import GridGameAi
from helper import LivePlotter
from pygame_training_ui import TrainingUI


@dataclass
class AlphaZeroTrainingConfig:
    """
    Small configuration object for AlphaZero training.

    The goal is readability first:
    you can open this file and immediately see the training settings in one place.
    """

    # --- Self-play length ---
    num_games: int = 50
    max_steps_per_game: int = 400

    # --- Network / optimizer ---
    learning_rate: float = 0.001

    # --- MCTS ---
    num_simulations: int = 25
    c_puct: float = 1.5
    root_dirichlet_alpha: float = 0.3
    root_dirichlet_epsilon: float = 0.25

    # --- Move selection ---
    # temperature controls exploration when sampling from MCTS visit counts:
    # e.g. 1.0 = more exploration, 0.0 = always choose the most visited action.
    temperature: float = 1.0
    temperature_drop_step: int = 10

    # --- UI / plotting ---
    use_training_ui: bool = False
    ui_show_every: int = 5
    ui_speed: int = 30
    use_live_plotter: bool = True


def build_environment(config: AlphaZeroTrainingConfig) -> GridGameAi:
    """
    Create the Quoridor environment and apply script-level settings.

    For now the environment already owns its reward parameters internally.
    We still set `max_steps_per_game` at the training-script level because
    it belongs more to the training loop than to the game rules themselves.
    """
    env = GridGameAi()
    return env


def build_agent(config: AlphaZeroTrainingConfig) -> AlphaZeroSelfPlayAgent:
    """
    Create the AlphaZero-style agent.

    This agent owns:
    - the policy-value network
    - the AlphaZero trainer
    - the self-play example buffer
    """
    return AlphaZeroSelfPlayAgent(
        lr=config.learning_rate,
        temperature=config.temperature,
    )


def build_ui_if_enabled(env: GridGameAi, config: AlphaZeroTrainingConfig):
    """
    Build the pygame UI only if explicitly enabled.

    This is useful because MCTS self-play can already be slow, so many runs are
    easier to do headless first.
    """
    if not config.use_training_ui:
        return None

    return TrainingUI(
        env=env,
        show_every=config.ui_show_every,
        speed=config.ui_speed,
    )


def build_plotter_if_enabled(config: AlphaZeroTrainingConfig):
    """
    Build the live matplotlib plotter only if enabled.
    """
    if not config.use_live_plotter:
        return None

    return LivePlotter()


def print_training_summary(config: AlphaZeroTrainingConfig):
    """
    Print a small readable summary before the run starts.
    """
    print("\n=== AlphaZero Training Configuration ===")
    print(f"Games: {config.num_games}")
    print(f"Max steps per game: {config.max_steps_per_game}")
    print(f"Learning rate: {config.learning_rate}")
    print(f"MCTS simulations per move: {config.num_simulations}")
    print(f"c_puct: {config.c_puct}")
    print(f"Dirichlet alpha: {config.root_dirichlet_alpha}")
    print(f"Dirichlet epsilon: {config.root_dirichlet_epsilon}")
    print(f"Temperature: {config.temperature}")
    print(f"Temperature drop step: {config.temperature_drop_step}")
    print(f"Use training UI: {config.use_training_ui}")
    print(f"Use live plotter: {config.use_live_plotter}")
    print("========================================\n")


def run_training(config: AlphaZeroTrainingConfig):
    """
    Create all main objects and launch the AlphaZero training loop.

    This function is intentionally small:
    the idea is that the script reads almost like a checklist.
    """
    env = build_environment(config)
    agent = build_agent(config)
    ui = build_ui_if_enabled(env, config)
    plotter = build_plotter_if_enabled(config)

    # The current training loop reads MAX_STEPS_PER_GAME from the game.agent module.
    # We override it here so this script stays the main source of truth.
    import agent as agent_module

    agent_module.MAX_STEPS_PER_GAME = config.max_steps_per_game

    train_alphazero_self_play(
        env=env,
        agent=agent,
        plotter=plotter,
        ui=ui,
        num_games=config.num_games,
        num_simulations=config.num_simulations,
        c_puct=config.c_puct,
        root_dirichlet_alpha=config.root_dirichlet_alpha,
        root_dirichlet_epsilon=config.root_dirichlet_epsilon,
        temperature=config.temperature,
        temperature_drop_step=config.temperature_drop_step,
    )


def main():
    """
    Main entrypoint for manual runs.

    If you want to experiment, the easiest workflow is:
    1. edit the config values below
    2. run this script again
    """
    config = AlphaZeroTrainingConfig(
        # Start with small values while debugging.
        # Later you can increase `num_games` and `num_simulations`.
        num_games=5,
        max_steps_per_game=400,
        learning_rate=0.001,
        num_simulations=10,
        c_puct=1.5,
        root_dirichlet_alpha=0.3,
        root_dirichlet_epsilon=0.25,
        temperature=1.0,
        temperature_drop_step=10,
        use_training_ui=False,
        ui_show_every=5,
        ui_speed=30,
        use_live_plotter=True,
    )

    print_training_summary(config)
    run_training(config)


if __name__ == "__main__":
    main()
