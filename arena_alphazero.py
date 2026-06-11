#!/usr/bin/env python3

"""
Visual arena for comparing two AlphaZero-style Quoridor checkpoints.

Examples:
    venv\\Scripts\\python.exe arena_alphazero.py --model-a path\\to\\best_a.pth --model-b path\\to\\best_b.pth
    venv\\Scripts\\python.exe arena_alphazero.py --model-a a_5x5.pth --model-b b_5x5.pth --num-games 4 --num-simulations 50
    venv\\Scripts\\python.exe arena_alphazero.py --board-size 9 --max-walls 10 --model-a old_9x9.pth --model-b old_9x9_b.pth
"""

from pathlib import Path
import argparse
from dataclasses import dataclass
import json
import sys

import torch


PROJECT_ROOT = Path(__file__).resolve().parent
GAME_DIR = PROJECT_ROOT / "game"
if str(GAME_DIR) not in sys.path:
    sys.path.insert(0, str(GAME_DIR))

from agent import AlphaZeroSelfPlayAgent  # noqa: E402
from game_logic_Ai import GridGameAi, NUM_MOVE_ACTIONS, P1, P2  # noqa: E402
from mcts import MCTS  # noqa: E402
from model import DEVICE  # noqa: E402


# ============================================================
# ARENA CONFIG
# ============================================================
# Edit these defaults if you prefer launching the arena with:
#     venv\\Scripts\\python.exe arena_alphazero.py

DEFAULT_MODEL_A = (
    PROJECT_ROOT
    / "quoridor_alphazero_runs"
    / "run_20260605_181100_balanced"
    / "models"
    / "latest_model.pth"
)
DEFAULT_MODEL_B = (
    PROJECT_ROOT
    / "quoridor_alphazero_runs"
    / "run_20260606_112507_2.5c_puct"
    / "models"
    / "latest_model.pth"
)
DEFAULT_NAME_A = "Agent A"
DEFAULT_NAME_B = "Agent B"
DEFAULT_BOARD_SIZE = 5
DEFAULT_MAX_WALLS = 4
DEFAULT_NUM_GAMES = 10
DEFAULT_MAX_STEPS = 70

# Per-agent search settings. These are the main knobs to edit before pressing Run.
# Example: set A to 10000 sims / c_puct 1.0 and B to 800 sims / c_puct 1.5.
DEFAULT_NUM_SIMULATIONS_A = 600
DEFAULT_NUM_SIMULATIONS_B = 600
DEFAULT_MCTS_BATCH_SIZE_A = 1
DEFAULT_MCTS_BATCH_SIZE_B = 1
DEFAULT_C_PUCT_A = 1.5
DEFAULT_C_PUCT_B = 2.5
DEFAULT_TEMPERATURE_A = 1
DEFAULT_TEMPERATURE_B = 1
DEFAULT_TEMPERATURE_AFTER_DROP_A = 0
DEFAULT_TEMPERATURE_AFTER_DROP_B = 0
DEFAULT_TEMPERATURE_DROP_STEP_A = 4
DEFAULT_TEMPERATURE_DROP_STEP_B = 4
DEFAULT_USE_UI = True
DEFAULT_UI_SPEED = 30


class ArenaUI:
    """
    Pygame arena view with the same visual style as play_vs_ai.py.

    The arena logic stays outside this class: this only renders the current
    environment, match metadata, and the running score.
    """

    BG = (248, 249, 250)
    PANEL_BG = (235, 238, 242)
    GRID_LINE = (145, 151, 160)
    TEXT = (24, 28, 33)
    MUTED = (86, 96, 110)
    BLUE = (38, 102, 220)
    RED = (215, 55, 65)
    GREEN = (41, 150, 85)

    def __init__(self, env, speed=30):
        import pygame

        self.pygame = pygame
        pygame.init()

        self.env = env
        self.speed = speed
        self.cell = 72 if env.grid_size <= 5 else 48
        self.board_margin = 120 if env.grid_size <= 5 else 48
        self.side_panel = 430
        self.board_pixels = self.cell * env.grid_size
        self.window_width = self.board_margin * 2 + self.board_pixels + self.side_panel
        self.window_height = max(self.board_margin * 2 + self.board_pixels, 600)
        self.panel_x = self.board_margin * 2 + self.board_pixels

        self.screen = pygame.display.set_mode((self.window_width, self.window_height))
        pygame.display.set_caption("Mini-Quoridor - AlphaZero Arena")
        self.clock = pygame.time.Clock()

        self.font_title = pygame.font.SysFont("arial", 28, bold=True)
        self.font_header = pygame.font.SysFont("arial", 22, bold=True)
        self.font_body = pygame.font.SysFont("arial", 18)
        self.font_small = pygame.font.SysFont("arial", 15)

        self.game_num = 0
        self.step = 0
        self.p1_name = "P1"
        self.p2_name = "P2"
        self.p1_config = None
        self.p2_config = None
        self.score = None
        self.last_message = "Arena ready."

    def new_game(self, game_num, p1_name, p2_name, p1_config, p2_config, score):
        self.game_num = game_num
        self.step = 0
        self.p1_name = p1_name
        self.p2_name = p2_name
        self.p1_config = p1_config
        self.p2_config = p2_config
        self.score = score
        self.last_message = "Starting new arena game."
        self.render()

    def render(self, step=None, message=None):
        if step is not None:
            self.step = step
        if message is not None:
            self.last_message = message

        for event in self.pygame.event.get():
            if event.type == self.pygame.QUIT:
                self.close()
                raise KeyboardInterrupt("Arena UI closed by user")

        self.screen.fill(self.BG)
        self._draw_grid()
        self._draw_walls()
        self._draw_players()
        self._draw_sidebar()
        self.pygame.display.flip()
        self.clock.tick(self.speed)

    def close(self):
        self.pygame.quit()

    def _draw_text(self, text, x, y, color=None, text_font=None):
        color = self.TEXT if color is None else color
        text_font = self.font_body if text_font is None else text_font
        surface = text_font.render(str(text), True, color)
        self.screen.blit(surface, (x, y))
        return surface.get_height()

    def _draw_wrapped_text(self, text, x, y, max_width, color=None, text_font=None):
        color = self.TEXT if color is None else color
        text_font = self.font_body if text_font is None else text_font
        words = str(text).split()
        lines = []
        current = ""

        for word in words:
            candidate = word if not current else f"{current} {word}"
            if text_font.size(candidate)[0] <= max_width:
                current = candidate
            else:
                if current:
                    lines.append(current)
                current = word

        if current:
            lines.append(current)

        for line in lines:
            surface = text_font.render(line, True, color)
            self.screen.blit(surface, (x, y))
            y += surface.get_height() + 3

        return y

    def _board_to_screen(self, row, col):
        return (
            self.board_margin + col * self.cell,
            self.board_margin + row * self.cell,
        )

    def _draw_grid(self):
        pygame = self.pygame
        board_rect = pygame.Rect(
            self.board_margin,
            self.board_margin,
            self.board_pixels,
            self.board_pixels,
        )
        pygame.draw.rect(self.screen, (255, 255, 255), board_rect)

        for row in range(self.env.grid_size):
            for col in range(self.env.grid_size):
                x, y = self._board_to_screen(row, col)
                rect = pygame.Rect(x, y, self.cell, self.cell)
                pygame.draw.rect(self.screen, (250, 250, 250), rect)
                pygame.draw.rect(self.screen, self.GRID_LINE, rect, 1)

        goal_top = pygame.Rect(
            self.board_margin,
            self.board_margin,
            self.board_pixels,
            5,
        )
        goal_bottom = pygame.Rect(
            self.board_margin,
            self.board_margin + self.board_pixels - 5,
            self.board_pixels,
            5,
        )
        pygame.draw.rect(self.screen, self.RED, goal_top)
        pygame.draw.rect(self.screen, self.BLUE, goal_bottom)

    def _draw_players(self):
        pygame = self.pygame
        players = [(P1, self.env.p1_pos, self.BLUE), (P2, self.env.p2_pos, self.RED)]

        for player, pos, color in players:
            cx = self.board_margin + int(pos[1]) * self.cell + self.cell // 2
            cy = self.board_margin + int(pos[0]) * self.cell + self.cell // 2
            pygame.draw.circle(self.screen, color, (cx, cy), self.cell // 3)
            label = self.font_header.render(str(player), True, (255, 255, 255))
            self.screen.blit(label, label.get_rect(center=(cx, cy)))

    def _draw_walls(self):
        pygame = self.pygame
        wall_specs = [
            (self.env.p1_horizontal_walls, "h", self.BLUE),
            (self.env.p1_vertical_walls, "v", self.BLUE),
            (self.env.p2_horizontal_walls, "h", self.RED),
            (self.env.p2_vertical_walls, "v", self.RED),
        ]

        for walls, orientation, color in wall_specs:
            for row, col in walls:
                if orientation == "h":
                    rect = pygame.Rect(
                        self.board_margin + col * self.cell,
                        self.board_margin + (row + 1) * self.cell - 6,
                        self.cell * 2,
                        12,
                    )
                else:
                    rect = pygame.Rect(
                        self.board_margin + (col + 1) * self.cell - 6,
                        self.board_margin + row * self.cell,
                        12,
                        self.cell * 2,
                    )
                pygame.draw.rect(self.screen, color, rect, border_radius=4)

    def _config_line(self, config):
        if config is None:
            return ""
        return (
            f"sims={config.num_simulations}, c={config.c_puct:g}, "
            f"temp={config.temperature:g}->{config.temperature_after_drop:g}"
            f"@{config.temperature_drop_step}"
        )

    def _draw_sidebar(self):
        pygame = self.pygame
        pygame.draw.rect(
            self.screen,
            self.PANEL_BG,
            pygame.Rect(self.panel_x, 0, self.side_panel, self.window_height),
        )

        x = self.panel_x + 18
        y = 24
        max_width = self.side_panel - 36

        self._draw_text("AlphaZero Arena", x, y, self.TEXT, self.font_header)
        y += 34
        self._draw_text(f"Game: {self.game_num}", x, y)
        y += 25
        self._draw_text(f"Step: {self.step}", x, y)
        y += 25
        self._draw_text(
            f"Board: {self.env.grid_size}x{self.env.grid_size}, walls={self.env.max_walls}",
            x,
            y,
        )
        y += 34

        self._draw_text("Current Players", x, y, self.TEXT, self.font_header)
        y += 29
        y = self._draw_wrapped_text(f"P1: {self.p1_name}", x, y, max_width, self.BLUE)
        y = self._draw_wrapped_text(
            self._config_line(self.p1_config),
            x,
            y,
            max_width,
            self.MUTED,
            self.font_small,
        )
        y += 8
        y = self._draw_wrapped_text(f"P2: {self.p2_name}", x, y, max_width, self.RED)
        y = self._draw_wrapped_text(
            self._config_line(self.p2_config),
            x,
            y,
            max_width,
            self.MUTED,
            self.font_small,
        )
        y += 22

        self._draw_text("Score", x, y, self.TEXT, self.font_header)
        y += 29
        if self.score is not None:
            self._draw_text(f"{self.score['name_a']}: {self.score['wins_a']}", x, y)
            y += 24
            self._draw_text(f"{self.score['name_b']}: {self.score['wins_b']}", x, y)
            y += 24
            self._draw_text(f"Draws/timeouts: {self.score['draws']}", x, y)
            y += 34

        self._draw_text("Walls", x, y, self.TEXT, self.font_header)
        y += 29
        self._draw_text(
            f"P1: {self.env.p1_available_walls}/{self.env.max_walls}", x, y, self.BLUE
        )
        y += 24
        self._draw_text(
            f"P2: {self.env.p2_available_walls}/{self.env.max_walls}", x, y, self.RED
        )
        y += 34

        turn_color = self.BLUE if self.env.turn == P1 else self.RED
        turn_name = self.p1_name if self.env.turn == P1 else self.p2_name
        y = self._draw_wrapped_text(
            f"Turn: P{self.env.turn} ({turn_name})",
            x,
            y,
            max_width,
            turn_color,
            self.font_body,
        )
        y += 22

        if self.last_message:
            self._draw_wrapped_text(
                self.last_message, x, y, max_width, self.TEXT, self.font_small
            )


@dataclass(frozen=True)
class ArenaAgentConfig:
    name: str
    num_simulations: int
    mcts_batch_size: int
    c_puct: float
    temperature: float
    temperature_after_drop: float
    temperature_drop_step: int

    def temperature_for_step(self, step_count):
        if self.temperature_drop_step <= 0:
            return self.temperature
        if step_count < self.temperature_drop_step:
            return self.temperature
        return self.temperature_after_drop


def infer_num_filters_from_state_dict(state_dict):
    first_conv = state_dict.get("trunk.0.block.0.weight")
    if first_conv is None:
        return 64
    return int(first_conv.shape[0])


def load_model_into_agent(checkpoint_path, temperature, board_size, max_walls):
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    loaded_checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    if isinstance(loaded_checkpoint, dict) and "model_state_dict" in loaded_checkpoint:
        checkpoint = loaded_checkpoint
        state_dict = checkpoint["model_state_dict"]
    else:
        checkpoint = {}
        state_dict = loaded_checkpoint

    num_filters = checkpoint.get(
        "num_filters",
        checkpoint.get("metadata", {}).get(
            "num_filters",
            infer_num_filters_from_state_dict(state_dict),
        ),
    )
    env_action_count = GridGameAi(
        grid_size=board_size, max_walls=max_walls
    ).total_actions
    agent = AlphaZeroSelfPlayAgent(
        lr=0.001,
        temperature=temperature,
        board_size=board_size,
        max_walls=max_walls,
        num_actions=env_action_count,
        num_filters=num_filters,
    )

    if checkpoint:
        checkpoint_board_size = checkpoint.get("board_size")
        checkpoint_max_walls = checkpoint.get("max_walls")
        if checkpoint_board_size is not None and checkpoint_board_size != board_size:
            raise ValueError(
                f"{checkpoint_path} was trained for {checkpoint_board_size}x"
                f"{checkpoint_board_size}, but arena uses {board_size}x{board_size}."
            )
        if checkpoint_max_walls is not None and checkpoint_max_walls != max_walls:
            raise ValueError(
                f"{checkpoint_path} was trained with {checkpoint_max_walls} walls, "
                f"but arena uses {max_walls}."
            )

    try:
        agent.model.load_state_dict(state_dict)
    except RuntimeError as exc:
        raise RuntimeError(
            f"Could not load {checkpoint_path}. Check that --board-size "
            f"({board_size}), --max-walls ({max_walls}), and num_filters "
            f"({num_filters}) match the checkpoint. The arena assumes the "
            f"default 3-convolution-block model architecture."
        ) from exc
    agent.model.to(DEVICE)
    agent.model.eval()
    return agent


def run_search(env, agent, num_simulations, mcts_batch_size, c_puct):
    search = MCTS(
        agent=agent,
        num_simulations=num_simulations,
        c_puct=c_puct,
        add_dirichlet_noise=False,
    )

    if mcts_batch_size > 1:
        return search, search.run_batched(env, batch_size=mcts_batch_size)

    return search, search.run(env)


def play_one_arena_game(
    env,
    p1_agent,
    p2_agent,
    p1_config,
    p2_config,
    ui,
    game_num,
    max_steps_per_game,
    score,
):
    env.reset()
    done = False
    step_count = 0
    wall_count = 0

    if ui is not None:
        ui.new_game(
            game_num=game_num,
            p1_name=p1_config.name,
            p2_name=p2_config.name,
            p1_config=p1_config,
            p2_config=p2_config,
            score=score,
        )

    while not done and step_count < max_steps_per_game:
        current_player = env.get_current_player()
        current_agent = p1_agent if current_player == P1 else p2_agent
        current_config = p1_config if current_player == P1 else p2_config

        if ui is not None:
            ui.render(
                step=step_count,
                message=(
                    f"{current_config.name} thinking "
                    f"({current_config.num_simulations} simulations)..."
                ),
            )

        search, root = run_search(
            env=env,
            agent=current_agent,
            num_simulations=current_config.num_simulations,
            mcts_batch_size=current_config.mcts_batch_size,
            c_puct=current_config.c_puct,
        )
        move_temperature = current_config.temperature_for_step(step_count)
        action = search.select_action(root, temperature=move_temperature)
        _, done, info = env.apply_action(action)

        if action >= NUM_MOVE_ACTIONS:
            wall_count += 1

        step_count += 1

        if ui is not None:
            action_type = "wall" if action >= NUM_MOVE_ACTIONS else "move"
            ui.render(
                step=step_count,
                message=f"{current_config.name} played a {action_type}.",
            )

        if info.get("invalid", False):
            raise RuntimeError(f"Arena selected invalid action {action}")

    timeout_reached = step_count >= max_steps_per_game and not env.is_terminal()
    winner = None if timeout_reached else env.get_winner()

    return {
        "winner": winner,
        "steps": step_count,
        "walls": wall_count,
        "timeout": timeout_reached,
    }


def model_name_for_winner(winner, p1_name, p2_name):
    if winner == P1:
        return p1_name
    if winner == P2:
        return p2_name
    return None


def build_agent_config(name, args, suffix):
    return ArenaAgentConfig(
        name=name,
        num_simulations=getattr(args, f"num_simulations_{suffix}"),
        mcts_batch_size=getattr(args, f"mcts_batch_size_{suffix}"),
        c_puct=getattr(args, f"c_puct_{suffix}"),
        temperature=getattr(args, f"temperature_{suffix}"),
        temperature_after_drop=getattr(args, f"temperature_after_drop_{suffix}"),
        temperature_drop_step=getattr(args, f"temperature_drop_step_{suffix}"),
    )


def agent_config_to_dict(config):
    return {
        "name": config.name,
        "num_simulations": config.num_simulations,
        "mcts_batch_size": config.mcts_batch_size,
        "c_puct": config.c_puct,
        "temperature": config.temperature,
        "temperature_after_drop": config.temperature_after_drop,
        "temperature_drop_step": config.temperature_drop_step,
    }


def play_match(args):
    env = GridGameAi(grid_size=args.board_size, max_walls=args.max_walls)
    model_a_config = build_agent_config(args.name_a, args, "a")
    model_b_config = build_agent_config(args.name_b, args, "b")

    model_a_agent = load_model_into_agent(
        args.model_a,
        temperature=model_a_config.temperature,
        board_size=args.board_size,
        max_walls=args.max_walls,
    )
    model_b_agent = load_model_into_agent(
        args.model_b,
        temperature=model_b_config.temperature,
        board_size=args.board_size,
        max_walls=args.max_walls,
    )

    ui = None
    if args.ui:
        ui = ArenaUI(env=env, speed=args.ui_speed)

    model_a_wins = 0
    model_b_wins = 0
    draws_or_timeouts = 0
    game_results = []

    for game_index in range(args.num_games):
        game_num = game_index + 1

        if game_index % 2 == 0:
            p1_agent = model_a_agent
            p2_agent = model_b_agent
            p1_config = model_a_config
            p2_config = model_b_config
            p1_name = args.name_a
            p2_name = args.name_b
        else:
            p1_agent = model_b_agent
            p2_agent = model_a_agent
            p1_config = model_b_config
            p2_config = model_a_config
            p1_name = args.name_b
            p2_name = args.name_a

        print(f"\n================ Arena Game {game_num} ================")
        print(
            f"P1: {p1_name} | sims={p1_config.num_simulations} | "
            f"c_puct={p1_config.c_puct} | temp={p1_config.temperature}->"
            f"{p1_config.temperature_after_drop}@{p1_config.temperature_drop_step} | "
            f"batch={p1_config.mcts_batch_size}"
        )
        print(
            f"P2: {p2_name} | sims={p2_config.num_simulations} | "
            f"c_puct={p2_config.c_puct} | temp={p2_config.temperature}->"
            f"{p2_config.temperature_after_drop}@{p2_config.temperature_drop_step} | "
            f"batch={p2_config.mcts_batch_size}"
        )
        score = {
            "name_a": args.name_a,
            "name_b": args.name_b,
            "wins_a": model_a_wins,
            "wins_b": model_b_wins,
            "draws": draws_or_timeouts,
        }

        result = play_one_arena_game(
            env=env,
            p1_agent=p1_agent,
            p2_agent=p2_agent,
            p1_config=p1_config,
            p2_config=p2_config,
            ui=ui,
            game_num=game_num,
            max_steps_per_game=args.max_steps,
            score=score,
        )

        winner_name = model_name_for_winner(result["winner"], p1_name, p2_name)
        if winner_name == args.name_a:
            model_a_wins += 1
        elif winner_name == args.name_b:
            model_b_wins += 1
        else:
            draws_or_timeouts += 1

        game_result = {
            "game": game_num,
            "p1_model": p1_name,
            "p2_model": p2_name,
            "p1_config": agent_config_to_dict(p1_config),
            "p2_config": agent_config_to_dict(p2_config),
            "winner": winner_name,
            "board_size": args.board_size,
            "max_walls": args.max_walls,
            "steps": result["steps"],
            "walls": result["walls"],
            "timeout": result["timeout"],
        }
        game_results.append(game_result)

        print(
            f"Winner: {winner_name or 'None'} | "
            f"steps={result['steps']} | walls={result['walls']} | "
            f"timeout={result['timeout']}"
        )
        if ui is not None:
            score = {
                "name_a": args.name_a,
                "name_b": args.name_b,
                "wins_a": model_a_wins,
                "wins_b": model_b_wins,
                "draws": draws_or_timeouts,
            }
            if result["timeout"]:
                message = "Game ended by timeout."
            elif winner_name is None:
                message = "Game ended in a draw."
            else:
                message = f"{winner_name} wins this game."
            ui.score = score
            ui.render(step=result["steps"], message=message)

    summary = {
        "model_a_name": args.name_a,
        "model_a_path": str(args.model_a),
        "model_b_name": args.name_b,
        "model_b_path": str(args.model_b),
        "device": str(DEVICE),
        "board_size": args.board_size,
        "max_walls": args.max_walls,
        "num_actions": env.total_actions,
        "num_games": args.num_games,
        "max_steps": args.max_steps,
        "model_a_config": agent_config_to_dict(model_a_config),
        "model_b_config": agent_config_to_dict(model_b_config),
        "model_a_wins": model_a_wins,
        "model_b_wins": model_b_wins,
        "draws_or_timeouts": draws_or_timeouts,
        "average_steps": (
            sum(result["steps"] for result in game_results) / len(game_results)
            if game_results
            else 0.0
        ),
        "games": game_results,
    }

    print("\n=== Arena Summary ===")
    print(f"{args.name_a} wins: {model_a_wins}")
    print(f"{args.name_b} wins: {model_b_wins}")
    print(f"Draws / timeouts: {draws_or_timeouts}")
    print(f"Average steps: {summary['average_steps']:.2f}")

    if args.summary_path:
        args.summary_path.parent.mkdir(parents=True, exist_ok=True)
        with args.summary_path.open("w", encoding="utf-8") as summary_file:
            json.dump(summary, summary_file, indent=2)
        print(f"Summary saved to: {args.summary_path}")

    if ui is not None:
        ui.close()

    return summary


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-a", type=Path, default=DEFAULT_MODEL_A)
    parser.add_argument("--model-b", type=Path, default=DEFAULT_MODEL_B)
    parser.add_argument("--name-a", default=DEFAULT_NAME_A)
    parser.add_argument("--name-b", default=DEFAULT_NAME_B)
    # Board variant:
    # - fast exam-friendly arena: board_size=5, max_walls=4
    # - classic Quoridor: pass --board-size 9 --max-walls 10
    parser.add_argument("--board-size", type=int, default=DEFAULT_BOARD_SIZE)
    parser.add_argument("--max-walls", type=int, default=DEFAULT_MAX_WALLS)
    parser.add_argument("--num-games", type=int, default=DEFAULT_NUM_GAMES)
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument(
        "--num-simulations",
        type=int,
        default=DEFAULT_NUM_SIMULATIONS_A,
        help="Shared MCTS simulations fallback for both agents.",
    )
    parser.add_argument(
        "--mcts-batch-size",
        type=int,
        default=DEFAULT_MCTS_BATCH_SIZE_A,
        help="Shared MCTS batch-size fallback for both agents.",
    )
    parser.add_argument(
        "--c-puct",
        type=float,
        default=DEFAULT_C_PUCT_A,
        help="Shared c_puct fallback for both agents.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE_A,
        help="Shared root temperature fallback for both agents.",
    )
    parser.add_argument(
        "--temperature-after-drop",
        type=float,
        default=DEFAULT_TEMPERATURE_AFTER_DROP_A,
        help="Shared root temperature after drop fallback for both agents.",
    )
    parser.add_argument(
        "--temperature-drop-step",
        type=int,
        default=DEFAULT_TEMPERATURE_DROP_STEP_A,
        help="Shared move step where root temperature drops. 0 disables drop.",
    )
    parser.add_argument(
        "--num-simulations-a",
        type=int,
        default=DEFAULT_NUM_SIMULATIONS_A,
        help="MCTS simulations for model A.",
    )
    parser.add_argument(
        "--num-simulations-b",
        type=int,
        default=DEFAULT_NUM_SIMULATIONS_B,
        help="MCTS simulations for model B.",
    )
    parser.add_argument(
        "--mcts-batch-size-a",
        type=int,
        default=DEFAULT_MCTS_BATCH_SIZE_A,
        help="MCTS batch size for model A.",
    )
    parser.add_argument(
        "--mcts-batch-size-b",
        type=int,
        default=DEFAULT_MCTS_BATCH_SIZE_B,
        help="MCTS batch size for model B.",
    )
    parser.add_argument(
        "--c-puct-a",
        type=float,
        default=DEFAULT_C_PUCT_A,
        help="PUCT exploration constant for model A.",
    )
    parser.add_argument(
        "--c-puct-b",
        type=float,
        default=DEFAULT_C_PUCT_B,
        help="PUCT exploration constant for model B.",
    )
    parser.add_argument(
        "--temperature-a",
        type=float,
        default=DEFAULT_TEMPERATURE_A,
        help="Root visit-count sampling temperature for model A.",
    )
    parser.add_argument(
        "--temperature-b",
        type=float,
        default=DEFAULT_TEMPERATURE_B,
        help="Root visit-count sampling temperature for model B.",
    )
    parser.add_argument(
        "--temperature-after-drop-a",
        type=float,
        default=DEFAULT_TEMPERATURE_AFTER_DROP_A,
        help="Root temperature for model A after its drop step.",
    )
    parser.add_argument(
        "--temperature-after-drop-b",
        type=float,
        default=DEFAULT_TEMPERATURE_AFTER_DROP_B,
        help="Root temperature for model B after its drop step.",
    )
    parser.add_argument(
        "--temperature-drop-step-a",
        type=int,
        default=DEFAULT_TEMPERATURE_DROP_STEP_A,
        help="Move step where model A drops temperature. 0 disables drop.",
    )
    parser.add_argument(
        "--temperature-drop-step-b",
        type=int,
        default=DEFAULT_TEMPERATURE_DROP_STEP_B,
        help="Move step where model B drops temperature. 0 disables drop.",
    )
    parser.add_argument("--ui-speed", type=int, default=DEFAULT_UI_SPEED)
    parser.add_argument(
        "--summary-path",
        type=Path,
        default=PROJECT_ROOT / "training_metrics" / "arena_summary.json",
    )
    parser.add_argument("--no-ui", dest="ui", action="store_false")
    parser.set_defaults(ui=DEFAULT_USE_UI)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.num_games <= 0:
        raise ValueError("--num-games must be positive")
    if args.max_steps <= 0:
        raise ValueError("--max-steps must be positive")
    if args.num_simulations <= 0:
        raise ValueError("--num-simulations must be positive")
    if args.num_simulations_a <= 0:
        raise ValueError("--num-simulations-a must be positive")
    if args.num_simulations_b <= 0:
        raise ValueError("--num-simulations-b must be positive")
    if args.mcts_batch_size <= 0:
        raise ValueError("--mcts-batch-size must be positive")
    if args.mcts_batch_size_a <= 0:
        raise ValueError("--mcts-batch-size-a must be positive")
    if args.mcts_batch_size_b <= 0:
        raise ValueError("--mcts-batch-size-b must be positive")
    if args.c_puct <= 0:
        raise ValueError("--c-puct must be positive")
    if args.c_puct_a <= 0:
        raise ValueError("--c-puct-a must be positive")
    if args.c_puct_b <= 0:
        raise ValueError("--c-puct-b must be positive")
    if args.temperature < 0:
        raise ValueError("--temperature cannot be negative")
    if args.temperature_a < 0:
        raise ValueError("--temperature-a cannot be negative")
    if args.temperature_b < 0:
        raise ValueError("--temperature-b cannot be negative")
    if args.temperature_after_drop < 0:
        raise ValueError("--temperature-after-drop cannot be negative")
    if args.temperature_after_drop_a < 0:
        raise ValueError("--temperature-after-drop-a cannot be negative")
    if args.temperature_after_drop_b < 0:
        raise ValueError("--temperature-after-drop-b cannot be negative")
    if args.temperature_drop_step < 0:
        raise ValueError("--temperature-drop-step cannot be negative")
    if args.temperature_drop_step_a < 0:
        raise ValueError("--temperature-drop-step-a cannot be negative")
    if args.temperature_drop_step_b < 0:
        raise ValueError("--temperature-drop-step-b cannot be negative")
    if args.board_size < 3:
        raise ValueError("--board-size must be at least 3")
    if args.max_walls < 0:
        raise ValueError("--max-walls cannot be negative")
    if args.max_walls > args.board_size * 2:
        raise ValueError("--max-walls looks too high for this board size")
    if not args.model_a.exists():
        raise FileNotFoundError(f"--model-a not found: {args.model_a}")
    if not args.model_b.exists():
        raise FileNotFoundError(f"--model-b not found: {args.model_b}")

    print("\n=== AlphaZero Arena Configuration ===")
    print(f"Device: {DEVICE}")
    print(f"Board: {args.board_size}x{args.board_size}")
    print(f"Max walls per player: {args.max_walls}")
    print(f"{args.name_a}: {args.model_a}")
    print(f"{args.name_b}: {args.model_b}")
    print(f"Games: {args.num_games}")
    print(f"Max steps: {args.max_steps}")
    print(
        f"{args.name_a} search: sims={args.num_simulations_a}, "
        f"c_puct={args.c_puct_a}, temp={args.temperature_a}->"
        f"{args.temperature_after_drop_a}@{args.temperature_drop_step_a}, "
        f"batch={args.mcts_batch_size_a}"
    )
    print(
        f"{args.name_b} search: sims={args.num_simulations_b}, "
        f"c_puct={args.c_puct_b}, temp={args.temperature_b}->"
        f"{args.temperature_after_drop_b}@{args.temperature_drop_step_b}, "
        f"batch={args.mcts_batch_size_b}"
    )
    print(f"Pygame UI: {args.ui}")
    print("=====================================\n")

    play_match(args)


if __name__ == "__main__":
    main()
