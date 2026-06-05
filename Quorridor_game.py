from pathlib import Path
import sys
import numpy as np
import pygame
import torch

# Setup path per importare da game/
PROJECT_ROOT = Path(__file__).resolve().parent
GAME_DIR = PROJECT_ROOT / "game"
if str(GAME_DIR) not in sys.path:
    sys.path.insert(0, str(GAME_DIR))

from game.game_logic_Ai import GridGameAi, NUM_MOVE_ACTIONS, MOVE_ACTIONS
from game.agent import AlphaZeroSelfPlayAgent
from game.mcts import MCTS

pygame.init()

# ============================================================
# PLAY AGAINST MODEL CONFIG
# ============================================================
# Edit these values to choose the model and variant you want to test.
#
# For the fast variant use BOARD_SIZE=5 and MAX_WALLS=4.
# For classic Quoridor use BOARD_SIZE=9 and MAX_WALLS=10, but only with
# checkpoints trained on the same board variant.
MODEL_PATH = (
    PROJECT_ROOT
    / "quoridor_alphazero_runs"
    / "run_20260603_134015"
    / "models"
    / "latest_model.pth"
)
MODEL_NAME = "latest_model"
BOARD_SIZE = 5
MAX_WALLS = 4
HUMAN_PLAYER = 1  # 1 = blue starts from bottom, 2 = red starts from top
AI_NUM_SIMULATIONS = 8000
AI_TEMPERATURE = 0.0
AI_C_PUCT = 1.5

# Board variant:
# - fast   play/demo: BOARD_SIZE=5, MAX_WALLS=4
# - classic Quoridor: BOARD_SIZE=9, MAX_WALLS=10
game = GridGameAi(grid_size=BOARD_SIZE, max_walls=MAX_WALLS)

CELL = 60
SIDE_PANEL = 330

W = CELL * game.grid_size + SIDE_PANEL
H = CELL * game.grid_size

screen = pygame.display.set_mode((W, H))
pygame.display.set_caption("Quoridor")

clock = pygame.time.Clock()

mode = "move"
ui_message1 = None

# ---------------------------
# UTILS
# ---------------------------


def cell_from_mouse(pos):
    screen_x, screen_y = pos
    # Convert screen coordinates to (row, col)
    row = screen_y // CELL
    col = screen_x // CELL
    return np.array([row, col])


def is_valid_wall_cell(row, col):
    return 0 <= row < game.grid_size - 1 and 0 <= col < game.grid_size - 1


def simulate_move(game, player, direction):
    current = game.p1_pos if player == 1 else game.p2_pos

    parts = direction.split("-")

    base = {
        "up": (-1, 0),
        "down": (1, 0),
        "left": (0, -1),
        "right": (0, 1),
    }

    drow, dcol = base[parts[0]]
    new_pos = current + np.array([drow, dcol])

    if len(parts) == 2 and parts[1] == "jump":
        new_pos = new_pos + np.array([drow, dcol])
    elif len(parts) == 2:
        sdrow, sdcol = base[parts[1]]
        new_pos = new_pos + np.array([sdrow, sdcol])

    return new_pos


# ---------------------------
# DRAW
# ---------------------------


def draw_grid():
    for row in range(game.grid_size):
        for col in range(game.grid_size):
            # Convert (row, col) to screen coordinates (screen_x, screen_y)
            screen_x = col * CELL
            screen_y = row * CELL
            rect = pygame.Rect(screen_x, screen_y, CELL, CELL)
            pygame.draw.rect(screen, (200, 200, 200), rect, 1)


def draw_players():
    colors = {1: (0, 0, 255), 2: (255, 0, 0)}

    for p, pos in [(1, game.p1_pos), (2, game.p2_pos)]:
        # pos is (row, col), convert to screen coordinates
        screen_x = pos[1] * CELL + CELL // 2
        screen_y = pos[0] * CELL + CELL // 2
        pygame.draw.circle(screen, colors[p], (screen_x, screen_y), CELL // 3)


def draw_walls():

    for row, col in game.p1_horizontal_walls:
        pygame.draw.rect(
            screen,
            (0, 0, 255),
            pygame.Rect(col * CELL, (row + 1) * CELL - 5, CELL * 2, 10),
        )
    for row, col in game.p2_horizontal_walls:
        pygame.draw.rect(
            screen,
            (255, 0, 0),
            pygame.Rect(col * CELL, (row + 1) * CELL - 5, CELL * 2, 10),
        )

    # Vertical walls: (row, col) format
    for row, col in game.p1_vertical_walls:
        pygame.draw.rect(
            screen,
            (0, 0, 255),
            pygame.Rect((col + 1) * CELL - 5, row * CELL, 10, CELL * 2),
        )

    for row, col in game.p2_vertical_walls:
        pygame.draw.rect(
            screen,
            (255, 0, 0),
            pygame.Rect((col + 1) * CELL - 5, row * CELL, 10, CELL * 2),
        )


# ---------------------------
# PREVIEW
# ---------------------------


def draw_move_preview():
    player = game.turn
    moves = game.available_moves(player)

    for _, direction in moves:
        pos = simulate_move(game, player, direction)

        # pos is (row, col), convert to screen coordinates
        screen_x = pos[1] * CELL + CELL // 2
        screen_y = pos[0] * CELL + CELL // 2
        pygame.draw.circle(
            screen,
            (0, 255, 0),
            (screen_x, screen_y),
            10,
        )


def draw_wall_preview(mouse_cell):
    row, col = mouse_cell

    if not is_valid_wall_cell(row, col):
        return  # non disegnare nulla fuori range

    keys = pygame.key.get_pressed()
    horizontal = keys[pygame.K_LSHIFT]

    if horizontal:
        rect = pygame.Rect(col * CELL, (row + 1) * CELL - 5, CELL * 2, 10)
    else:
        rect = pygame.Rect((col + 1) * CELL - 5, row * CELL, 10, CELL * 2)

    pygame.draw.rect(screen, (0, 255, 0), rect, 2)


def draw_sidebar():
    pygame.draw.rect(
        screen, (240, 240, 240), pygame.Rect(CELL * game.grid_size, 0, SIDE_PANEL, H)
    )

    font = pygame.font.SysFont(None, 24)

    txt1 = font.render("Press 'M' to switch to move mode", True, (0, 0, 0))
    txt2 = font.render("Press 'W' to switch to wall mode", True, (0, 0, 0))
    txt3 = font.render("Hold 'LShift' to change wall orientation", True, (0, 0, 0))
    txt4 = font.render(
        f"P1 walls: {game.p1_available_walls}/{game.max_walls}",
        True,
        (0, 0, 200),
    )
    txt5 = font.render(
        f"P2 walls: {game.p2_available_walls}/{game.max_walls}",
        True,
        (200, 0, 0),
    )
    txt6 = font.render(
        f" Turn: player {'blue' if game.turn == 1 else 'red'}",
        True,
        (0, 0, 255) if game.turn == 1 else (255, 0, 0),
    )
    txt7 = font.render(
        f"Human: P{HUMAN_PLAYER} | AI: P{ai_player}",
        True,
        (0, 0, 0),
    )
    txt8 = font.render(f"Model: {MODEL_NAME}", True, (0, 0, 0))
    txt9 = font.render(f"AI sims: {AI_NUM_SIMULATIONS}", True, (0, 0, 0))

    screen.blit(txt1, (CELL * game.grid_size + 10, 20))
    screen.blit(txt2, (CELL * game.grid_size + 10, 50))
    screen.blit(txt3, (CELL * game.grid_size + 10, 80))
    screen.blit(txt4, (CELL * game.grid_size + 10, 110))
    screen.blit(txt5, (CELL * game.grid_size + 10, 140))
    screen.blit(txt6, (CELL * game.grid_size + 10, 170))
    screen.blit(txt7, (CELL * game.grid_size + 10, 200))
    screen.blit(txt8, (CELL * game.grid_size + 10, 230))
    screen.blit(txt9, (CELL * game.grid_size + 10, 260))

    if ui_message1:
        msg_surface = font.render(ui_message1, True, (0, 0, 0))
        screen.blit(msg_surface, (CELL * game.grid_size + 10, 290))
    if winner:
        font_big = pygame.font.SysFont(None, 36)
        txt = font_big.render(f"P{winner} WINS!", True, (0, 220, 0))
        screen.blit(txt, (CELL * game.grid_size + 10, 320))


# ---------------------------
# MODEL
# ---------------------------


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
                f"but this demo uses BOARD_SIZE={board_size}."
            )
        if checkpoint_max_walls is not None and checkpoint_max_walls != max_walls:
            raise ValueError(
                f"Checkpoint max_walls={checkpoint_max_walls}, "
                f"but this demo uses MAX_WALLS={max_walls}."
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
    agent = AlphaZeroSelfPlayAgent(
        lr=0.001,
        temperature=0.0,
        board_size=board_size,
        max_walls=max_walls,
        num_actions=num_actions,
        num_filters=num_filters,
    )

    try:
        agent.model.load_state_dict(state_dict)
    except RuntimeError as exc:
        raise RuntimeError(
            "Could not load the checkpoint. Make sure BOARD_SIZE and MAX_WALLS "
            f"match the model you trained. Inferred num_filters={num_filters}."
        ) from exc
    agent.model.eval()
    return agent


def is_human_turn():
    return game.turn == HUMAN_PLAYER


def reset_match():
    global winner, ui_message1, mode
    game.reset()
    winner = None
    ui_message1 = None
    mode = "move"


def play_ai_turn(agent):
    search = MCTS(
        agent=agent,
        num_simulations=AI_NUM_SIMULATIONS,
        c_puct=AI_C_PUCT,
        add_dirichlet_noise=False,
    )
    root = search.run(game)
    action = search.select_action(root, temperature=AI_TEMPERATURE)
    _, _, info = game.apply_action(action)
    if info.get("invalid", False):
        raise RuntimeError(f"AI selected invalid action {action}")
    return action


# ---------------------------
# GAME LOOP
# ---------------------------
if __name__ == "__main__":
    if HUMAN_PLAYER not in (1, 2):
        raise ValueError("HUMAN_PLAYER must be 1 or 2")

    ai_player = 2 if HUMAN_PLAYER == 1 else 1
    agent = None

    print("Loading model...")
    print(f"Model: {MODEL_NAME}")
    print(f"Path: {MODEL_PATH}")
    print(f"Board: {BOARD_SIZE}x{BOARD_SIZE} | Max walls: {MAX_WALLS}")
    print(f"Total action slots: {game.total_actions}")
    print(f"Human player: P{HUMAN_PLAYER} | AI player: P{ai_player}")
    agent = load_model_into_agent(
        MODEL_PATH,
        board_size=BOARD_SIZE,
        max_walls=MAX_WALLS,
    )
    print("Model loaded. Starting game against AI.\n")

    running = True
    winner = None

    while running:
        screen.fill((255, 255, 255))

        mouse_pos = pygame.mouse.get_pos()
        mouse_cell = cell_from_mouse(mouse_pos)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_m:
                    mode = "move"
                elif event.key == pygame.K_w:
                    mode = "wall"
                elif event.key == pygame.K_r:
                    reset_match()

            if event.type == pygame.MOUSEBUTTONDOWN:
                if winner is not None:
                    continue  # victory will stop further actions

                player = game.turn

                if not is_human_turn():
                    continue

                # ---------------- MOVE ----------------
                if mode == "move":
                    moves = game.available_moves(player)

                    for _, direction in moves:
                        new_pos = simulate_move(game, player, direction)

                        if np.array_equal(new_pos, mouse_cell):
                            game.move(player, ("move", direction))
                            winner = game.check_winner()
                            break

                # ---------------- WALL ----------------
                elif mode == "wall":
                    row, col = mouse_cell

                    if not is_valid_wall_cell(row, col):
                        continue  # blocca click invalidi

                    horizontal = pygame.key.get_pressed()[pygame.K_LSHIFT]
                    orientation = "h" if horizontal else "v"

                    success, message = game.place_wall(player, (row, col), orientation)
                    ui_message1 = message

                    if success:
                        mode = "move"
                        winner = game.check_winner()

        # DRAW
        draw_grid()
        draw_walls()
        draw_players()
        draw_sidebar()

        # Draw preview solo se la partita non è finita e se è il turno di un umano (in modalità 2)
        if winner is None and is_human_turn():
            if mode == "move":
                draw_move_preview()
            else:
                draw_wall_preview(mouse_cell)

        pygame.display.flip()
        clock.tick(60)

        # ---------------- AI TURN ----------------
        if not is_human_turn() and winner is None:
            action = play_ai_turn(agent)
            winner = game.check_winner()
            action_type = "wall" if action >= NUM_MOVE_ACTIONS else "move"
            ui_message1 = f"AI played {action_type}"

    pygame.quit()
