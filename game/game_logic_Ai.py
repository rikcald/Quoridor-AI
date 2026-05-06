import numpy as np
from collections import deque

P1 = 1
P2 = 2

MOVE_ACTIONS = [
    "up",
    "down",
    "left",
    "right",
    "up-jump",
    "down-jump",
    "left-jump",
    "right-jump",
    "up-left",
    "up-right",
    "down-left",
    "down-right",
    "left-up",
    "left-down",
    "right-up",
    "right-down",
]
NUM_MOVE_ACTIONS = len(MOVE_ACTIONS)  # 16
NUM_WALL_POS = (9 - 1) ** 2  # = 64

TOTAL_ACTIONS = NUM_MOVE_ACTIONS + 2 * NUM_WALL_POS  # 144


class GridGameAi:
    def __init__(self, grid_size=9):
        self.grid_size = grid_size
        self.reset()

    def reset(self):
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=int)

        self.p1_pos = np.array([self.grid_size - 1, self.grid_size // 2])
        self.p2_pos = np.array([0, self.grid_size // 2])  # top middle

        self.grid[self.p1_pos[0], self.p1_pos[1]] = P1
        self.grid[self.p2_pos[0], self.p2_pos[1]] = P2

        self.horizontal_walls = set()
        self.vertical_walls = set()

        self.p1_available_walls = 10
        self.p2_available_walls = 10

        self.turn = P1

    # ------------ Reinforcement Learning Methods ------------

    def decode_action(self, action):
        if action < NUM_MOVE_ACTIONS:
            return "move", MOVE_ACTIONS[action]
        elif action < NUM_MOVE_ACTIONS + NUM_WALL_POS:
            wall_index = action - NUM_MOVE_ACTIONS  # so it becames 0-63
            row = wall_index // (self.grid_size - 1)
            col = wall_index % (self.grid_size - 1)
            return "wall", (row, col), "h"
        else:
            wall_index = action - NUM_MOVE_ACTIONS - NUM_WALL_POS  # so it becames 0-63
            row = wall_index // (self.grid_size - 1)
            col = wall_index % (self.grid_size - 1)
            return "wall", (row, col), "v"

    def encode_action(self, action_type, *args):
        if action_type == "move":
            direction = args[0]
            return MOVE_ACTIONS.index(direction)
        elif action_type == "wall":
            row, col = args[0]
            orientation = args[1]
            if orientation == "h":
                wall_index = row * (self.grid_size - 1) + col
                return NUM_MOVE_ACTIONS + wall_index
            else:  # orientation == "v"
                wall_index = row * (self.grid_size - 1) + col
                return NUM_MOVE_ACTIONS + NUM_WALL_POS + wall_index

    def step(self, action):
        player = self.turn
        decoded = self.decode_action(action)

        invalid = False

        # --- esegui azione ---
        if decoded[0] == "move":
            _, direction = decoded

            try:
                self.move(player, ("move", direction))
            except ValueError:
                invalid = True

        else:
            _, location, orientation = decoded
            success, _ = self.place_wall(player, location, orientation)

            if not success:
                invalid = True

        # --- reward per azione invalida ---
        if invalid:
            return self.get_state(), -0.1, False, {"invalid": True}

        # --- controlla fine partita ---
        winner = self.check_winner()
        done = winner is not None

        # --- reward ---
        if done:
            reward = 1 if winner == player else -1
        else:
            reward = -0.01

        # --- stato ---
        state = self.get_state()

        return state, reward, done, {}

    def get_state(self):
        size = self.grid_size

        state = np.zeros((7, size, size), dtype=np.float32)

        # --- player 1 ---
        state[0, self.p1_pos[0], self.p1_pos[1]] = 1.0

        # --- player 2 ---
        state[1, self.p2_pos[0], self.p2_pos[1]] = 1.0

        # --- muri orizzontali ---
        for r, c in self.horizontal_walls:
            state[2, r, c] = 1.0

        # --- muri verticali ---
        for r, c in self.vertical_walls:
            state[3, r, c] = 1.0

        # --- muri rimanenti (broadcast) ---
        state[4, :, :] = self.p1_available_walls / 10.0
        state[5, :, :] = self.p2_available_walls / 10.0

        # --- turno --- broadcast: 1.0 se è il turno di P1, -1.0 se è il turno di P2
        state[6, :, :] = 1.0 if self.turn == P1 else -1.0

        return state

    # returns a binary mask of shape (TOTAL_ACTIONS,) where valid actions are 1.0 and invalid actions are 0.0 e.g. [0, 1, 0, ..., 1]
    def get_action_mask(self, player):
        mask = np.zeros(TOTAL_ACTIONS, dtype=np.float32)

        # --- MOVIMENTI ---
        moves = self.available_moves(player)

        # Set mask[id] = 1.0 for valid move actions
        for move in moves:
            action_id = self.encode_action(move)
            mask[action_id] = 1.0

        # --- MURI ---
        for row in range(self.grid_size - 1):
            for col in range(self.grid_size - 1):
                # orizzontale
                if self._is_valid_wall(player, (row, col), "h"):
                    action_id = self.encode_action("wall", (row, col), "h")
                    mask[action_id] = 1.0

                # verticale
                if self._is_valid_wall(player, (row, col), "v"):
                    action_id = self.encode_action("wall", (row, col), "v")
                    mask[action_id] = 1.0

        return mask

    def random_agent_action(env, player):
        mask = env.get_action_mask(player)

        valid_actions = np.where(mask == 1)[0]

        if len(valid_actions) == 0:
            raise Exception("No valid actions available")

        return np.random.choice(valid_actions)

    # ---------------------------------------------------

    def move(self, player, move):

        if player != self.turn:
            print("Non è il tuo turno")
            return

        moves = self.available_moves(player)

        if move not in moves:
            raise ValueError(f"Mossa non valida: {move}")

        action, direction = move

        if player == P1:
            current = self.p1_pos.copy()
        else:
            current = self.p2_pos.copy()

        # parsing direzione
        parts = direction.split("-")

        # movimento base (row, col) coordinates
        base_dirs = {
            "up": (-1, 0),
            "down": (1, 0),
            "left": (0, -1),
            "right": (0, 1),
        }

        drow, dcol = base_dirs[parts[0]]

        new_pos = current + np.array([drow, dcol])

        # caso jump
        if len(parts) == 2 and parts[1] == "jump":
            new_pos = new_pos + np.array([drow, dcol])

        # caso diagonale
        elif len(parts) == 2:
            sdrow, sdcol = base_dirs[parts[1]]
            new_pos = new_pos + np.array([sdrow, sdcol])

        # aggiorna griglia
        self.grid[current[0], current[1]] = 0
        self.grid[new_pos[0], new_pos[1]] = player

        if player == P1:
            self.p1_pos = new_pos
        else:
            self.p2_pos = new_pos

        print(f"Player {player} moves {new_pos} via {direction}")
        self.turn = P2 if self.turn == P1 else P1
        return direction

    def check_winner(self):

        if self.p1_pos[0] == 0:  # row == 0 (top)
            return P1
        elif self.p2_pos[0] == self.grid_size - 1:  # row == grid_size-1 (bottom)
            return P2
        return None

    # Expired: this is now handled in pygame_ui.py
    def print_grid(self):
        print("-" * 30)

        for row in range(self.grid_size):
            # riga celle
            line = ""
            for col in range(self.grid_size):
                line += str(self.grid[row, col])

                # muro verticale
                if col < self.grid_size - 1:
                    if (row, col) in self.vertical_walls:
                        line += "|"
                    else:
                        line += " "
            print(line)

            # riga muri orizzontali
            if row < self.grid_size - 1:
                line = ""
                for col in range(self.grid_size):
                    if (row, col) in self.horizontal_walls:
                        line += "──"
                    else:
                        line += "  "
                print(line)

        print("-" * 30)

    def available_moves(self, player):
        if player == P1:
            current = self.p1_pos
            opponent = self.p2_pos
        else:
            current = self.p2_pos
            opponent = self.p1_pos

        moves = []

        directions = {
            "up": (-1, 0),
            "down": (1, 0),
            "left": (0, -1),
            "right": (0, 1),
        }

        for d, (drow, dcol) in directions.items():
            adj = current + np.array([drow, dcol])

            if not self.is_inside_grid(adj):
                continue

            # caso normale: cella libera
            if (
                not np.array_equal(adj, opponent)
                and self.grid[adj[0], adj[1]] == 0
                and not self.is_blocked(current, adj)
            ):
                moves.append(("move", d))
                continue

            # caso salto
            if np.array_equal(adj, opponent) and not self.is_blocked(current, adj):
                jump = adj + np.array([drow, dcol])

                if (
                    self.is_inside_grid(jump)
                    and self.grid[jump[0], jump[1]] == 0
                    and not self.is_blocked(adj, jump)
                ):
                    moves.append(("move", f"{d}-jump"))
                else:
                    # diagonali
                    if d in ["up", "down"]:
                        sides = [("left", (0, -1)), ("right", (0, 1))]
                    else:
                        sides = [("up", (-1, 0)), ("down", (1, 0))]

                    for sd, (sdrow, sdcol) in sides:
                        diag = adj + np.array([sdrow, sdcol])

                        if (
                            self.is_inside_grid(diag)
                            and self.grid[diag[0], diag[1]] == 0
                            and not self.is_blocked(adj, diag)
                        ):
                            moves.append(("move", f"{d}-{sd}"))

        return moves

    def is_inside_grid(self, pos):
        pos_array = np.atleast_1d(pos)
        return 0 <= pos_array[0] < self.grid_size and 0 <= pos_array[1] < self.grid_size

    def place_wall(self, player, location, orientation):
        if player != self.turn:
            print("Non è il tuo turno")
            return False, None

        row, col = location

        # Validare il muro
        if not self._is_valid_wall(player, location, orientation):
            print("Posizionamento muro non valido")
            return False, "Invalid placement!"

        # Piazzare il muro
        if orientation == "h":
            self.horizontal_walls.add((row, col))
        else:
            self.vertical_walls.add((row, col))

        # Decrementare i muri disponibili
        if player == P1:
            self.p1_available_walls -= 1
        else:
            self.p2_available_walls -= 1

        self.turn = P2 if self.turn == P1 else P1
        return True, None

    def _is_valid_wall(self, player, location, orientation):
        """Valida se un muro può essere piazzato senza modificare lo stato."""
        row, col = location

        # Controllare se il giocatore ha muri disponibili
        if player == P1 and self.p1_available_walls <= 0:
            return False
        if player == P2 and self.p2_available_walls <= 0:
            return False

        # Controllare i limiti (muri stanno tra le celle)
        if row < 0 or col < 0 or row >= self.grid_size - 1 or col >= self.grid_size - 1:
            return False

        # Controllare collisioni con altri muri
        if orientation == "h":
            if (
                (row, col) in self.horizontal_walls
                or (row, col) in self.vertical_walls
                or (row, col + 1) in self.horizontal_walls
                or (row, col - 1) in self.horizontal_walls
            ):
                return False
        elif orientation == "v":
            if (
                (row, col) in self.vertical_walls
                or (row, col) in self.horizontal_walls
                or (row + 1, col) in self.vertical_walls
                or (row - 1, col) in self.vertical_walls
            ):
                return False
        else:
            return False

        # Controllare che il muro non blocchi completamente i percorsi
        # Simulare il piazzamento temporaneo
        if orientation == "h":
            self.horizontal_walls.add((row, col))
        else:
            self.vertical_walls.add((row, col))

        # Verificare se entrambi i giocatori hanno ancora un percorso
        valid = self._players_have_path()

        # Rollback della simulazione
        if orientation == "h":
            self.horizontal_walls.remove((row, col))
        else:
            self.vertical_walls.remove((row, col))

        return valid

    def is_blocked(self, a, b):
        row1, col1 = a
        row2, col2 = b

        # movimento verticale (up/down)
        if col1 == col2:
            row = min(row1, row2)

            # muro orizzontale blocca
            if (row, col1) in self.horizontal_walls:
                return True
            if (row, col1 - 1) in self.horizontal_walls:
                return True

        # movimento orizzontale (left/right)
        elif row1 == row2:
            col = min(col1, col2)

            # muro verticale blocca
            if (row1, col) in self.vertical_walls:
                return True
            if (row1 - 1, col) in self.vertical_walls:
                return True

        return False

    # BFS per verificare se c'è un percorso valido per entrambi i giocatori dopo il posizionamento del muro
    def _has_path(self, start, target_row):
        visited = set()
        queue = deque([tuple(start)])

        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        while queue:
            row, col = queue.popleft()

            # raggiunto il goal
            if row == target_row:
                return True

            for drow, dcol in directions:
                nrow, ncol = row + drow, col + dcol
                next_pos = (nrow, ncol)

                if not self.is_inside_grid(next_pos):
                    continue

                if next_pos in visited:
                    continue

                # muro blocca?
                if self.is_blocked((row, col), next_pos):
                    continue

                visited.add(next_pos)
                queue.append(next_pos)

        return False

    def _players_have_path(self):
        p1_ok = self._has_path(
            self.p1_pos, target_row=0
        )  # P1 needs to reach top (row 0)
        p2_ok = self._has_path(
            self.p2_pos, target_row=self.grid_size - 1
        )  # P2 needs to reach bottom
        return p1_ok and p2_ok
