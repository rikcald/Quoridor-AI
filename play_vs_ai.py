from pathlib import Path
import sys

import numpy as np
import pygame
import torch


PROJECT_ROOT = Path(__file__).resolve().parent
GAME_DIR = PROJECT_ROOT / "game"
if str(GAME_DIR) not in sys.path:
    sys.path.insert(0, str(GAME_DIR))

from agent import AlphaZeroSelfPlayAgent  # noqa: E402
from game_logic_Ai import GridGameAi, NUM_MOVE_ACTIONS  # noqa: E402
from mcts import MCTS  # noqa: E402


# ============================================================
# PLAY AGAINST MODEL CONFIG
# ============================================================
# Put the checkpoint you want to submit here. The default points to the best
# current 5x5 Mini-Quoridor run. If you move the model, only edit this path.
MODEL_PATH = (
    PROJECT_ROOT
    / "quoridor_alphazero_runs"
    / "run_20260606_112507_2.5c_puct_best"
    / "models"
    / "latest_model.pth"
)
MODEL_NAME = "AlphaZero Mini-Quoridor"

BOARD_SIZE = 5
MAX_WALLS = 4
HUMAN_PLAYER = 1
AI_C_PUCT = 2.5
AI_TEMPERATURE = 0.0
MAX_STEPS_PER_GAME = 120

DIFFICULTIES = [
    ("Easy", 20, "Fast search, useful for a quick demo."),
    ("Medium", 200, "Similar scale to early training games."),
    ("Challenging", 1000, "Stronger and slower."),
    ("Hard", 10000, "Very slow, strongest search budget."),
]

CELL = 72
BOARD_MARGIN = 200
SIDE_PANEL = 390
BOARD_PIXELS = CELL * BOARD_SIZE
WINDOW_WIDTH = BOARD_MARGIN * 2 + BOARD_PIXELS + SIDE_PANEL
WINDOW_HEIGHT = max(BOARD_MARGIN * 2 + BOARD_PIXELS, 560)
PANEL_X = BOARD_MARGIN * 2 + BOARD_PIXELS

BG = (248, 249, 250)
PANEL_BG = (235, 238, 242)
GRID_LINE = (145, 151, 160)
TEXT = (24, 28, 33)
MUTED = (86, 96, 110)
BLUE = (38, 102, 220)
RED = (215, 55, 65)
GREEN = (41, 150, 85)
WALL_PREVIEW = (34, 180, 95)
BUTTON = (255, 255, 255)
BUTTON_HOVER = (229, 240, 255)
BUTTON_BORDER = (154, 163, 178)


pygame.init()
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("Mini-Quoridor - AlphaZero Demo")
clock = pygame.time.Clock()

game = GridGameAi(grid_size=BOARD_SIZE, max_walls=MAX_WALLS)
ai_player = 2 if HUMAN_PLAYER == 1 else 1

agent = None
selected_difficulty = None
ai_num_simulations = None
mode = "move"
wall_orientation = "v"
winner = None
step_count = 0
move_limit_reached = False
ui_message = "Choose a difficulty to start."


def font(size, bold=False):
    return pygame.font.SysFont("arial", size, bold=bold)


FONT_TITLE = font(28, bold=True)
FONT_HEADER = font(22, bold=True)
FONT_BODY = font(18)
FONT_SMALL = font(15)
FONT_TINY = font(13)


def draw_text(text, x, y, color=TEXT, text_font=FONT_BODY):
    surface = text_font.render(str(text), True, color)
    screen.blit(surface, (x, y))
    return surface.get_height()


def draw_wrapped_text(
    text, x, y, max_width, color=TEXT, text_font=FONT_BODY, line_gap=3
):
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
        screen.blit(surface, (x, y))
        y += surface.get_height() + line_gap

    return y


def board_rect():
    return pygame.Rect(BOARD_MARGIN, BOARD_MARGIN, BOARD_PIXELS, BOARD_PIXELS)


def board_to_screen(row, col):
    return (
        BOARD_MARGIN + col * CELL,
        BOARD_MARGIN + row * CELL,
    )


def cell_from_mouse(pos):
    x, y = pos
    if not board_rect().collidepoint(x, y):
        return None
    col = (x - BOARD_MARGIN) // CELL
    row = (y - BOARD_MARGIN) // CELL
    if row < 0 or row >= game.grid_size or col < 0 or col >= game.grid_size:
        return None
    return np.array([row, col])


def is_valid_wall_cell(row, col):
    return 0 <= row < game.grid_size - 1 and 0 <= col < game.grid_size - 1


def simulate_move(player, direction):
    current = game.p1_pos if player == 1 else game.p2_pos
    base = {
        "up": (-1, 0),
        "down": (1, 0),
        "left": (0, -1),
        "right": (0, 1),
    }

    parts = direction.split("-")
    drow, dcol = base[parts[0]]
    new_pos = current + np.array([drow, dcol])

    if len(parts) == 2 and parts[1] == "jump":
        new_pos = new_pos + np.array([drow, dcol])
    elif len(parts) == 2:
        sdrow, sdcol = base[parts[1]]
        new_pos = new_pos + np.array([sdrow, sdcol])

    return new_pos


def infer_num_filters_from_state_dict(state_dict):
    first_conv = state_dict.get("trunk.0.block.0.weight")
    if first_conv is None:
        return 64
    return int(first_conv.shape[0])


def load_model_into_agent(checkpoint_path, board_size, max_walls):
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    loaded_checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(loaded_checkpoint, dict) and "model_state_dict" in loaded_checkpoint:
        checkpoint = loaded_checkpoint
        state_dict = checkpoint["model_state_dict"]
        checkpoint_board_size = checkpoint.get("board_size")
        checkpoint_max_walls = checkpoint.get("max_walls")

        if checkpoint_board_size is not None and checkpoint_board_size != board_size:
            raise ValueError(
                f"Checkpoint board_size={checkpoint_board_size}, "
                f"but the demo uses BOARD_SIZE={board_size}."
            )
        if checkpoint_max_walls is not None and checkpoint_max_walls != max_walls:
            raise ValueError(
                f"Checkpoint max_walls={checkpoint_max_walls}, "
                f"but the demo uses MAX_WALLS={max_walls}."
            )
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
    num_actions = GridGameAi(grid_size=board_size, max_walls=max_walls).total_actions
    loaded_agent = AlphaZeroSelfPlayAgent(
        lr=0.001,
        temperature=0.0,
        board_size=board_size,
        max_walls=max_walls,
        num_actions=num_actions,
        num_filters=num_filters,
    )
    loaded_agent.model.load_state_dict(state_dict)
    loaded_agent.model.eval()
    return loaded_agent


def is_human_turn():
    return game.turn == HUMAN_PLAYER


def reset_match():
    global winner, step_count, mode, wall_orientation, move_limit_reached, ui_message
    game.reset()
    winner = None
    step_count = 0
    move_limit_reached = False
    mode = "move"
    wall_orientation = "v"
    ui_message = "New game. Your turn." if is_human_turn() else "New game. AI starts."


def start_game(difficulty):
    global agent, selected_difficulty, ai_num_simulations, ui_message

    selected_difficulty = difficulty
    ai_num_simulations = difficulty[1]
    ui_message = "Loading model..."
    draw_scene()
    pygame.display.flip()

    if agent is None:
        agent = load_model_into_agent(
            MODEL_PATH,
            board_size=BOARD_SIZE,
            max_walls=MAX_WALLS,
        )

    reset_match()
    ui_message = f"{selected_difficulty[0]} selected. Your turn."


def play_ai_turn():
    global ui_message, step_count

    ui_message = f"AI thinking ({ai_num_simulations} simulations)..."
    draw_scene()
    pygame.display.flip()

    search = MCTS(
        agent=agent,
        num_simulations=ai_num_simulations,
        c_puct=AI_C_PUCT,
        add_dirichlet_noise=False,
    )
    root = search.run(game)
    action = search.select_action(root, temperature=AI_TEMPERATURE)
    _, _, info = game.apply_action(action)

    if info.get("invalid", False):
        raise RuntimeError(f"AI selected invalid action {action}")

    step_count += 1
    action_type = "wall" if action >= NUM_MOVE_ACTIONS else "move"
    ui_message = f"AI played a {action_type}."


def get_result_text():
    if move_limit_reached and winner is None:
        return "Draw"
    if winner is None:
        return None
    if winner == HUMAN_PLAYER:
        return "Human wins"
    return "AI wins"


def check_game_end():
    global winner, move_limit_reached, ui_message

    winner = game.check_winner()
    if winner is not None:
        ui_message = "Game finished."
        return

    if step_count < MAX_STEPS_PER_GAME:
        return

    move_limit_reached = True
    winner = game.get_timeout_adjudication_winner()
    if winner is None:
        ui_message = "Move limit reached. The game is a draw."
    else:
        ui_message = "Move limit reached. Closest legal path wins."


def draw_grid():
    pygame.draw.rect(screen, (255, 255, 255), board_rect())

    for row in range(game.grid_size):
        for col in range(game.grid_size):
            x, y = board_to_screen(row, col)
            rect = pygame.Rect(x, y, CELL, CELL)
            pygame.draw.rect(screen, (250, 250, 250), rect)
            pygame.draw.rect(screen, GRID_LINE, rect, 1)

    goal_top = pygame.Rect(BOARD_MARGIN, BOARD_MARGIN, BOARD_PIXELS, 5)
    goal_bottom = pygame.Rect(
        BOARD_MARGIN,
        BOARD_MARGIN + BOARD_PIXELS - 5,
        BOARD_PIXELS,
        5,
    )
    pygame.draw.rect(screen, RED, goal_top)
    pygame.draw.rect(screen, BLUE, goal_bottom)


def draw_players():
    players = [(1, game.p1_pos, BLUE), (2, game.p2_pos, RED)]

    for player, pos, color in players:
        cx = BOARD_MARGIN + int(pos[1]) * CELL + CELL // 2
        cy = BOARD_MARGIN + int(pos[0]) * CELL + CELL // 2
        pygame.draw.circle(screen, color, (cx, cy), CELL // 3)
        label = FONT_HEADER.render(str(player), True, (255, 255, 255))
        screen.blit(label, label.get_rect(center=(cx, cy)))


def draw_walls():
    wall_specs = [
        (game.p1_horizontal_walls, "h", BLUE),
        (game.p1_vertical_walls, "v", BLUE),
        (game.p2_horizontal_walls, "h", RED),
        (game.p2_vertical_walls, "v", RED),
    ]

    for walls, orientation, color in wall_specs:
        for row, col in walls:
            if orientation == "h":
                rect = pygame.Rect(
                    BOARD_MARGIN + col * CELL,
                    BOARD_MARGIN + (row + 1) * CELL - 6,
                    CELL * 2,
                    12,
                )
            else:
                rect = pygame.Rect(
                    BOARD_MARGIN + (col + 1) * CELL - 6,
                    BOARD_MARGIN + row * CELL,
                    12,
                    CELL * 2,
                )
            pygame.draw.rect(screen, color, rect, border_radius=4)


def draw_move_preview():
    player = game.turn
    for _, direction in game.available_moves(player):
        pos = simulate_move(player, direction)
        cx = BOARD_MARGIN + int(pos[1]) * CELL + CELL // 2
        cy = BOARD_MARGIN + int(pos[0]) * CELL + CELL // 2
        pygame.draw.circle(screen, GREEN, (cx, cy), 10)


def draw_wall_preview(mouse_cell):
    if mouse_cell is None:
        return

    row, col = mouse_cell
    if not is_valid_wall_cell(int(row), int(col)):
        return

    if wall_orientation == "h":
        rect = pygame.Rect(
            BOARD_MARGIN + int(col) * CELL,
            BOARD_MARGIN + (int(row) + 1) * CELL - 6,
            CELL * 2,
            12,
        )
    else:
        rect = pygame.Rect(
            BOARD_MARGIN + (int(col) + 1) * CELL - 6,
            BOARD_MARGIN + int(row) * CELL,
            12,
            CELL * 2,
        )

    pygame.draw.rect(screen, WALL_PREVIEW, rect, width=3, border_radius=4)


def draw_button(rect, title, subtitle, selected=False):
    mouse_over = rect.collidepoint(pygame.mouse.get_pos())
    color = BUTTON_HOVER if mouse_over or selected else BUTTON
    pygame.draw.rect(screen, color, rect, border_radius=8)
    pygame.draw.rect(screen, BUTTON_BORDER, rect, width=1, border_radius=8)
    draw_text(title, rect.x + 14, rect.y + 10, TEXT, FONT_HEADER)
    draw_wrapped_text(
        subtitle, rect.x + 14, rect.y + 39, rect.width - 28, MUTED, FONT_SMALL
    )


def difficulty_buttons():
    buttons = []
    x = BOARD_MARGIN
    y = 185
    width = WINDOW_WIDTH - BOARD_MARGIN * 2
    height = 72

    for idx, difficulty in enumerate(DIFFICULTIES):
        rect = pygame.Rect(x, y + idx * (height + 12), width, height)
        buttons.append((rect, difficulty))

    return buttons


def draw_menu():
    screen.fill(BG)
    draw_text("Mini-Quoridor", BOARD_MARGIN, 34, TEXT, FONT_TITLE)
    draw_text(
        "Play against an AlphaZero-style agent", BOARD_MARGIN, 68, MUTED, FONT_HEADER
    )
    y = 112
    y = draw_wrapped_text(
        "Choose how many MCTS simulations the model can use for each move. "
        "Higher difficulty means stronger search, but the AI will think longer.",
        BOARD_MARGIN,
        y,
        WINDOW_WIDTH - BOARD_MARGIN * 2,
        TEXT,
        FONT_BODY,
    )
    # y += 18
    # draw_text("Difficulty", BOARD_MARGIN, y, TEXT, FONT_HEADER)

    for index, (rect, difficulty) in enumerate(difficulty_buttons(), start=1):
        name, simulations, description = difficulty
        title = f"{index}. {name} - {simulations} simulations"
        draw_button(rect, title, description)

    footer_y = WINDOW_HEIGHT - 44
    draw_text(
        "Click a difficulty or press 1-4. Press Esc to quit.",
        BOARD_MARGIN,
        footer_y,
        MUTED,
        FONT_SMALL,
    )


def draw_sidebar():
    pygame.draw.rect(
        screen,
        PANEL_BG,
        pygame.Rect(PANEL_X, 0, SIDE_PANEL, WINDOW_HEIGHT),
    )

    x = PANEL_X + 18
    y = 24
    max_width = SIDE_PANEL - 36

    draw_text("AlphaZero Demo", x, y, TEXT, FONT_HEADER)
    y += 34
    y = draw_wrapped_text(MODEL_NAME, x, y, max_width, MUTED, FONT_SMALL)
    y += 14

    difficulty_name = selected_difficulty[0] if selected_difficulty else "-"
    draw_text(f"Difficulty: {difficulty_name}", x, y, TEXT, FONT_BODY)
    y += 25
    draw_text(f"AI simulations: {ai_num_simulations}", x, y, TEXT, FONT_BODY)
    y += 25
    draw_text(f"Step: {step_count}/{MAX_STEPS_PER_GAME}", x, y, TEXT, FONT_BODY)
    y += 34

    draw_text("Players", x, y, TEXT, FONT_HEADER)
    y += 29
    draw_text(f"Human: P{HUMAN_PLAYER} (blue)", x, y, BLUE, FONT_BODY)
    y += 24
    draw_text(f"AI: P{ai_player} (red)", x, y, RED, FONT_BODY)
    y += 34

    draw_text("Walls", x, y, TEXT, FONT_HEADER)
    y += 29
    draw_text(f"P1: {game.p1_available_walls}/{game.max_walls}", x, y, BLUE, FONT_BODY)
    y += 24
    draw_text(f"P2: {game.p2_available_walls}/{game.max_walls}", x, y, RED, FONT_BODY)
    y += 34

    turn_color = BLUE if game.turn == 1 else RED
    turn_owner = "Human" if is_human_turn() else "AI"
    draw_text(f"Turn: P{game.turn} ({turn_owner})", x, y, turn_color, FONT_BODY)
    y += 34

    draw_text("Controls", x, y, TEXT, FONT_HEADER)
    y += 28
    controls = [
        "M: move mode",
        "W: wall mode",
        "H/V or Space: wall direction",
        "R: restart",
        "Esc: quit",
    ]
    for item in controls:
        draw_text(item, x, y, MUTED, FONT_SMALL)
        y += 20

    y += 8
    draw_text(f"Mode: {mode}", x, y, TEXT, FONT_BODY)
    y += 24
    draw_text(
        f"Wall: {'horizontal' if wall_orientation == 'h' else 'vertical'}",
        x,
        y,
        TEXT,
        FONT_BODY,
    )
    y += 32

    if ui_message:
        y = draw_wrapped_text(ui_message, x, y, max_width, TEXT, FONT_SMALL)

    result_text = get_result_text()
    if result_text:
        result_color = GREEN if winner == HUMAN_PLAYER else RED
        y = max(y + 20, WINDOW_HEIGHT - 110)
        draw_text(result_text, x, y, result_color, FONT_TITLE)
        y += 36
        draw_text("Press R to play again.", x, y, MUTED, FONT_SMALL)


def draw_scene():
    if selected_difficulty is None:
        draw_menu()
        return

    screen.fill(BG)
    draw_grid()
    draw_walls()
    draw_players()

    mouse_cell = cell_from_mouse(pygame.mouse.get_pos())
    if winner is None and is_human_turn():
        if mode == "move":
            draw_move_preview()
        else:
            draw_wall_preview(mouse_cell)

    draw_sidebar()


def try_human_move(mouse_cell):
    global winner, step_count, ui_message

    if mouse_cell is None:
        return

    player = game.turn
    for _, direction in game.available_moves(player):
        new_pos = simulate_move(player, direction)
        if np.array_equal(new_pos, mouse_cell):
            game.move(player, ("move", direction))
            step_count += 1
            ui_message = "Move played."
            check_game_end()
            return

    ui_message = "Click one of the green legal moves."


def try_human_wall(mouse_cell):
    global mode, winner, step_count, ui_message

    if mouse_cell is None:
        return

    row, col = mouse_cell
    if not is_valid_wall_cell(int(row), int(col)):
        ui_message = "Wall must start inside a valid wall cell."
        return

    success, message = game.place_wall(
        game.turn,
        (int(row), int(col)),
        wall_orientation,
    )

    if success:
        step_count += 1
        mode = "move"
        ui_message = "Wall placed."
        check_game_end()
    else:
        ui_message = message or "Invalid wall placement."


def handle_menu_click(pos):
    for rect, difficulty in difficulty_buttons():
        if rect.collidepoint(pos):
            start_game(difficulty)
            return


def select_difficulty_by_key(key):
    index_by_key = {
        pygame.K_1: 0,
        pygame.K_2: 1,
        pygame.K_3: 2,
        pygame.K_4: 3,
    }
    if key in index_by_key:
        start_game(DIFFICULTIES[index_by_key[key]])


def handle_events():
    global mode, wall_orientation, running

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
            return

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False
                return

            if selected_difficulty is None:
                select_difficulty_by_key(event.key)
                continue

            if event.key == pygame.K_m:
                mode = "move"
            elif event.key == pygame.K_w:
                mode = "wall"
            elif event.key == pygame.K_h:
                wall_orientation = "h"
            elif event.key == pygame.K_v:
                wall_orientation = "v"
            elif event.key == pygame.K_SPACE:
                wall_orientation = "h" if wall_orientation == "v" else "v"
            elif event.key == pygame.K_r:
                reset_match()

        if event.type == pygame.MOUSEBUTTONDOWN:
            if selected_difficulty is None:
                handle_menu_click(event.pos)
                continue

            if winner is not None or not is_human_turn():
                continue

            mouse_cell = cell_from_mouse(event.pos)
            if mode == "move":
                try_human_move(mouse_cell)
            else:
                try_human_wall(mouse_cell)


def maybe_play_ai():
    global ui_message

    if selected_difficulty is None:
        return
    if winner is not None or move_limit_reached or is_human_turn():
        return
    if step_count >= MAX_STEPS_PER_GAME:
        ui_message = "Move limit reached. Restart with R."
        return

    play_ai_turn()
    check_game_end()


if __name__ == "__main__":
    if HUMAN_PLAYER not in (1, 2):
        raise ValueError("HUMAN_PLAYER must be 1 or 2")

    print("Mini-Quoridor AlphaZero demo")
    print(f"Model path: {MODEL_PATH}")
    print(f"Board: {BOARD_SIZE}x{BOARD_SIZE} | Max walls: {MAX_WALLS}")
    print("Choose difficulty in the Pygame window.")

    running = True
    while running:
        handle_events()
        maybe_play_ai()
        draw_scene()
        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
