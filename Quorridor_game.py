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

# Board variant:
# - fast exam-friendly play/demo: BOARD_SIZE=5, MAX_WALLS=4
# - classic Quoridor: BOARD_SIZE=9, MAX_WALLS=10
BOARD_SIZE = 5
MAX_WALLS = 4

# Use a checkpoint trained with the same BOARD_SIZE and MAX_WALLS.
MODEL_PATH = "model/PolicyValueNet_alphazero_latest.pth"

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

    screen.blit(txt1, (CELL * game.grid_size + 10, 20))
    screen.blit(txt2, (CELL * game.grid_size + 10, 50))
    screen.blit(txt3, (CELL * game.grid_size + 10, 80))
    screen.blit(txt4, (CELL * game.grid_size + 10, 110))
    screen.blit(txt5, (CELL * game.grid_size + 10, 140))
    screen.blit(txt6, (CELL * game.grid_size + 10, 170))

    if ui_message1:
        msg_surface = font.render(ui_message1, True, (0, 0, 0))
        screen.blit(msg_surface, (CELL * game.grid_size + 10, 180))
    if winner:
        font_big = pygame.font.SysFont(None, 36)
        txt = font_big.render(f"P{winner} WINS!", True, (0, 220, 0))
        screen.blit(txt, (CELL * game.grid_size + 10, 200))


# ---------------------------
# MODEL
# ---------------------------


def load_model_into_agent(checkpoint_path, board_size, max_walls):

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    num_actions = GridGameAi(grid_size=board_size, max_walls=max_walls).total_actions
    agent = AlphaZeroSelfPlayAgent(
        lr=0.001,
        temperature=0.0,
        board_size=board_size,
        max_walls=max_walls,
        num_actions=num_actions,
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
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
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    try:
        agent.model.load_state_dict(state_dict)
    except RuntimeError as exc:
        raise RuntimeError(
            "Could not load the checkpoint. Make sure BOARD_SIZE and MAX_WALLS "
            "match the model you trained."
        ) from exc
    agent.model.eval()
    return agent


# ---------------------------
# GAME LOOP
# ---------------------------
if __name__ == "__main__":
    # game_mode = "1"  # classica
    # game_mode = "2"  # umano vs IA
    game_mode = "2"
    num_simulations = 50
    agent = None
    if game_mode == "2":
        print("Loading model...")
        print(f"Board: {BOARD_SIZE}x{BOARD_SIZE} | Max walls: {MAX_WALLS}")
        print(f"Total action slots: {game.total_actions}")
        agent = load_model_into_agent(
            MODEL_PATH,
            board_size=BOARD_SIZE,
            max_walls=MAX_WALLS,
        )
        print("Model loaded. Starting game against AI (you are P1 - blue).\n")

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

            if event.type == pygame.MOUSEBUTTONDOWN:
                if winner is not None:
                    continue  # victory will stop further actions

                player = game.turn

                # Solo l'umano può cliccare, o in modalità "1" qualsiasi giocatore
                if game_mode == "2" and player == 2:
                    continue  # L'IA sta per giocare, ignora click umani

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
        if winner is None and (
            game_mode == "1" or (game_mode == "2" and game.turn == 1)
        ):
            if mode == "move":
                draw_move_preview()
            else:
                draw_wall_preview(mouse_cell)

        pygame.display.flip()
        clock.tick(60)

        # ---------------- AI TURN ----------------
        if game_mode == "2" and game.turn == 2 and winner is None:
            search = MCTS(
                agent=agent,
                num_simulations=num_simulations,
                c_puct=1.5,
                add_dirichlet_noise=False,
            )
            root = search.run(game)
            action = search.select_action(root, temperature=0.0)

            _, done, info = game.apply_action(action)
            winner = game.check_winner()
            ui_message1 = "AI played!"

    pygame.quit()
