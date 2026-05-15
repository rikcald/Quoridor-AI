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


def flip_direction_for_p2(direction):
    # We normalize P2 with a vertical board flip, so only the row axis changes:
    # e.g. "up" -> "down", "down-jump" -> "up-jump", "left" stays "left".
    parts = direction.split("-")
    flipped_parts = []

    for part in parts:
        if part == "up":
            flipped_parts.append("down")
        elif part == "down":
            flipped_parts.append("up")
        else:
            flipped_parts.append(part)

    return "-".join(flipped_parts)


class GridGameAi:
    def __init__(self, grid_size=9):
        self.grid_size = grid_size
        self.p1_reward_history = deque(maxlen=5)
        self.p2_reward_history = deque(maxlen=5)
        self.reset()

    def reset(self):
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=int)

        self.p1_pos = np.array([self.grid_size - 1, self.grid_size // 2])
        self.p2_pos = np.array([0, self.grid_size // 2])

        self.grid[self.p1_pos[0], self.p1_pos[1]] = P1
        self.grid[self.p2_pos[0], self.p2_pos[1]] = P2

        self.p1_horizontal_walls = set()
        self.p1_vertical_walls = set()
        self.p2_horizontal_walls = set()
        self.p2_vertical_walls = set()

        self.p1_available_walls = 10
        self.p2_available_walls = 10

        self.turn = P1

        self.p1_reward_history.clear()
        self.p2_reward_history.clear()

    def _flip_wall_row_for_p2(self, row):
        # Wall coordinates live on an 8x8 wall grid for a 9x9 board, so P2's
        # canonical view mirrors only the wall row index: e.g. row 0 -> 7, row 2 -> 5.
        return (self.grid_size - 2) - row

    def _transform_wall_row(self, row, player):
        # Helper used when building the neural-network state.
        # P1 keeps the real coordinates, P2 sees the board vertically flipped.
        if player == P1:
            return row
        return self._flip_wall_row_for_p2(row)

    # ------------ Reinforcement Learning Methods ------------

    def decode_action(self, action, player=None):
        if player is None:
            player = self.turn

        if action < NUM_MOVE_ACTIONS:
            # The network always predicts actions in canonical coordinates.
            # e.g. action "up" means "move toward the opponent side" for both players,
            # so for P2 we must map it back to the real-board move "down".
            direction = MOVE_ACTIONS[action]
            if player == P2:
                direction = flip_direction_for_p2(direction)
            return "move", direction
        elif action < NUM_MOVE_ACTIONS + NUM_WALL_POS:
            wall_index = action - NUM_MOVE_ACTIONS
            row = wall_index // (self.grid_size - 1)
            col = wall_index % (self.grid_size - 1)
            # Wall actions are also stored in canonical coordinates:
            # e.g. a wall predicted on canonical row 0 for P2 must be placed on real row 7.
            if player == P2:
                row = self._flip_wall_row_for_p2(row)
            return "wall", (row, col), "h"
        else:
            wall_index = action - NUM_MOVE_ACTIONS - NUM_WALL_POS
            row = wall_index // (self.grid_size - 1)
            col = wall_index % (self.grid_size - 1)
            if player == P2:
                row = self._flip_wall_row_for_p2(row)
            return "wall", (row, col), "v"

    def encode_action(self, action_type, *args, player=None):
        if player is None:
            player = self.turn

        if action_type == "move":
            # Inverse of decode_action(): real-board moves are converted to canonical
            # action indices before touching the policy/Q output space.
            # e.g. P2 real move "down" becomes canonical move "up".
            direction = args[0]
            if player == P2:
                direction = flip_direction_for_p2(direction)
            return MOVE_ACTIONS.index(direction)
        elif action_type == "wall":
            row, col = args[0]
            orientation = args[1]
            # Same idea for walls: the action head should see a single shared indexing
            # scheme for both sides, so P2 wall rows are mirrored before indexing.
            if player == P2:
                row = self._flip_wall_row_for_p2(row)
            if orientation == "h":
                wall_index = row * (self.grid_size - 1) + col
                return NUM_MOVE_ACTIONS + wall_index
            else:
                wall_index = row * (self.grid_size - 1) + col
                return NUM_MOVE_ACTIONS + NUM_WALL_POS + wall_index
        raise ValueError(f"Invalid action encoding: {action_type}, {args}")

    def step(self, action):
        player = self.turn
        decoded = self.decode_action(action, player=player)

        invalid = False

        p1_old_pos = self.p1_pos.copy()
        p2_old_pos = self.p2_pos.copy()

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

        if invalid:
            return self.get_state(), -0.1, False, {"invalid": True}

        winner = self.check_winner()
        done = winner is not None

        if done:
            reward = 20 if winner == player else -20

        else:
            if decoded[0] == "move":
                if player == P1:
                    reward = (p1_old_pos[0] - self.p1_pos[0]) * 0.1
                else:
                    reward = (self.p2_pos[0] - p2_old_pos[0]) * 0.1
            else:
                reward = -0.01

            if player == P1:
                self.p1_reward_history.append(reward)
                if (
                    len(self.p1_reward_history) >= 5
                    and sum(self.p1_reward_history) == 0
                ):
                    reward -= 0.3
            else:
                self.p2_reward_history.append(reward)
                if (
                    len(self.p2_reward_history) >= 5
                    and sum(self.p2_reward_history) == 0
                ):
                    reward -= 0.3

        state = self.get_state()

        return state, reward, done, {"invalid": False, "winner": winner}

    def get_state_player_centric(self, player):
        size = self.grid_size
        # Canonical self-play state with 6 planes:
        # 0 = me, 1 = opponent, 2 = horizontal walls, 3 = vertical walls,
        # 4 = my remaining walls, 5 = opponent remaining walls.
        # The old turn plane is not needed because plane 0 is always the current player.
        state = np.zeros((6, size, size), dtype=np.float32)

        if player == P1:
            my_pos = self.p1_pos
            opp_pos = self.p2_pos
            my_horizontal_walls = self.p1_horizontal_walls
            opp_horizontal_walls = self.p2_horizontal_walls
            my_vertical_walls = self.p1_vertical_walls
            opp_vertical_walls = self.p2_vertical_walls
            my_available_walls = self.p1_available_walls
            opp_available_walls = self.p2_available_walls
            row_transform = lambda row: row
        else:
            my_pos = self.p2_pos
            opp_pos = self.p1_pos
            my_horizontal_walls = self.p2_horizontal_walls
            opp_horizontal_walls = self.p1_horizontal_walls
            my_vertical_walls = self.p2_vertical_walls
            opp_vertical_walls = self.p1_vertical_walls
            my_available_walls = self.p2_available_walls
            opp_available_walls = self.p1_available_walls
            # P2 is mirrored only on rows, not on columns:
            # e.g. real pawn at (0, 4) is seen by P2 as (8, 4).
            row_transform = lambda row: size - 1 - row

        # Plane 0 is always "the player whose turn generated this state".
        state[0, row_transform(my_pos[0]), my_pos[1]] = 1.0
        # Plane 1 is always the opponent in the same canonical orientation.
        state[1, row_transform(opp_pos[0]), opp_pos[1]] = 1.0

        for r, c in my_horizontal_walls:
            state[2, self._transform_wall_row(r, player), c] = 1.0
        for r, c in opp_horizontal_walls:
            state[2, self._transform_wall_row(r, player), c] = 1.0

        for r, c in my_vertical_walls:
            state[3, self._transform_wall_row(r, player), c] = 1.0
        for r, c in opp_vertical_walls:
            state[3, self._transform_wall_row(r, player), c] = 1.0

        state[4, :, :] = my_available_walls / 10.0
        state[5, :, :] = opp_available_walls / 10.0

        return state

    def get_state(self):
        # During self-play/training we always expose the board from the perspective
        # of the player to move, so both sides feed the same representation style
        # into a shared model.
        return self.get_state_player_centric(self.turn)

    def get_action_mask(self, player):
        mask = np.zeros(TOTAL_ACTIONS, dtype=np.float32)

        moves = self.available_moves(player)

        for move in moves:
            # Valid real-board moves are converted into canonical action ids.
            action_id = self.encode_action(*move, player=player)
            mask[action_id] = 1.0

        for row in range(self.grid_size - 1):
            for col in range(self.grid_size - 1):
                if self._is_valid_wall(player, (row, col), "h"):
                    # This keeps the mask aligned with decode_action() for both players:
                    # e.g. a valid P2 wall on real row 7 activates the canonical row-0 action id.
                    action_id = self.encode_action(
                        "wall", (row, col), "h", player=player
                    )
                    mask[action_id] = 1.0

                if self._is_valid_wall(player, (row, col), "v"):
                    action_id = self.encode_action(
                        "wall", (row, col), "v", player=player
                    )
                    mask[action_id] = 1.0

        return mask

    def move(self, player, move):
        if player != self.turn:
            print("Non e il tuo turno")
            return

        moves = self.available_moves(player)

        if move not in moves:
            raise ValueError(f"Mossa non valida: {move}")

        _, direction = move

        if player == P1:
            current = self.p1_pos.copy()
        else:
            current = self.p2_pos.copy()

        parts = direction.split("-")

        base_dirs = {
            "up": (-1, 0),
            "down": (1, 0),
            "left": (0, -1),
            "right": (0, 1),
        }

        drow, dcol = base_dirs[parts[0]]
        new_pos = current + np.array([drow, dcol])

        if len(parts) == 2 and parts[1] == "jump":
            new_pos = new_pos + np.array([drow, dcol])
        elif len(parts) == 2:
            sdrow, sdcol = base_dirs[parts[1]]
            new_pos = new_pos + np.array([sdrow, sdcol])

        self.grid[current[0], current[1]] = 0
        self.grid[new_pos[0], new_pos[1]] = player

        if player == P1:
            self.p1_pos = new_pos
        else:
            self.p2_pos = new_pos

        self.turn = P2 if self.turn == P1 else P1
        return direction

    def check_winner(self):
        if self.p1_pos[0] == 0:
            return P1
        elif self.p2_pos[0] == self.grid_size - 1:
            return P2
        return None

    def print_grid(self):
        print("-" * 30)

        for row in range(self.grid_size):
            line = ""
            for col in range(self.grid_size):
                line += str(self.grid[row, col])

                if col < self.grid_size - 1:
                    if (row, col) in self.p1_vertical_walls or (
                        row,
                        col,
                    ) in self.p2_vertical_walls:
                        line += "|"
                    else:
                        line += " "
            print(line)

            if row < self.grid_size - 1:
                line = ""
                for col in range(self.grid_size):
                    if (row, col) in self.p1_horizontal_walls or (
                        row,
                        col,
                    ) in self.p2_horizontal_walls:
                        line += "--"
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

            if (
                not np.array_equal(adj, opponent)
                and self.grid[adj[0], adj[1]] == 0
                and not self.is_blocked(current, adj)
            ):
                moves.append(("move", d))
                continue

            if np.array_equal(adj, opponent) and not self.is_blocked(current, adj):
                jump = adj + np.array([drow, dcol])

                if (
                    self.is_inside_grid(jump)
                    and self.grid[jump[0], jump[1]] == 0
                    and not self.is_blocked(adj, jump)
                ):
                    moves.append(("move", f"{d}-jump"))
                else:
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
            print("Non e il tuo turno")
            return False, None

        row, col = location

        if not self._is_valid_wall(player, location, orientation):
            print("Posizionamento muro non valido")
            print(
                f"Player {player} attempted to place a wall at {location} with orientation {orientation}, but it was invalid."
            )
            return False, "Invalid placement!"

        if orientation == "h":
            if player == P1:
                self.p1_horizontal_walls.add((row, col))
            else:
                self.p2_horizontal_walls.add((row, col))
        else:
            if player == P1:
                self.p1_vertical_walls.add((row, col))
            else:
                self.p2_vertical_walls.add((row, col))

        if player == P1:
            self.p1_available_walls -= 1
        else:
            self.p2_available_walls -= 1

        self.turn = P2 if self.turn == P1 else P1
        return True, None

    def _is_valid_wall(self, player, location, orientation):
        row, col = location

        if player == P1 and self.p1_available_walls <= 0:
            return False
        if player == P2 and self.p2_available_walls <= 0:
            return False

        if row < 0 or col < 0 or row >= self.grid_size - 1 or col >= self.grid_size - 1:
            return False

        if orientation == "h":
            if (
                (row, col) in self.p1_horizontal_walls
                or (row, col) in self.p2_horizontal_walls
                or (row, col) in self.p1_vertical_walls
                or (row, col) in self.p2_vertical_walls
                or (row, col + 1) in self.p1_horizontal_walls
                or (row, col + 1) in self.p2_horizontal_walls
                or (row, col - 1) in self.p1_horizontal_walls
                or (row, col - 1) in self.p2_horizontal_walls
            ):
                return False
        elif orientation == "v":
            if (
                (row, col) in self.p1_vertical_walls
                or (row, col) in self.p2_vertical_walls
                or (row, col) in self.p1_horizontal_walls
                or (row, col) in self.p2_horizontal_walls
                or (row + 1, col) in self.p1_vertical_walls
                or (row + 1, col) in self.p2_vertical_walls
                or (row - 1, col) in self.p1_vertical_walls
                or (row - 1, col) in self.p2_vertical_walls
            ):
                return False
        else:
            return False

        if orientation == "h":
            if player == P1:
                self.p1_horizontal_walls.add((row, col))
            else:
                self.p2_horizontal_walls.add((row, col))
        else:
            if player == P1:
                self.p1_vertical_walls.add((row, col))
            else:
                self.p2_vertical_walls.add((row, col))

        valid = self._players_have_path()

        if orientation == "h":
            if player == P1:
                self.p1_horizontal_walls.remove((row, col))
            else:
                self.p2_horizontal_walls.remove((row, col))
        else:
            if player == P1:
                self.p1_vertical_walls.remove((row, col))
            else:
                self.p2_vertical_walls.remove((row, col))

        return valid

    def is_blocked(self, a, b):
        row1, col1 = a
        row2, col2 = b

        if col1 == col2:
            row = min(row1, row2)

            if (row, col1) in self.p1_horizontal_walls or (
                row,
                col1,
            ) in self.p2_horizontal_walls:
                return True
            if (row, col1 - 1) in self.p1_horizontal_walls or (
                row,
                col1 - 1,
            ) in self.p2_horizontal_walls:
                return True

        elif row1 == row2:
            col = min(col1, col2)

            if (row1, col) in self.p1_vertical_walls or (
                row1,
                col,
            ) in self.p2_vertical_walls:
                return True
            if (row1 - 1, col) in self.p1_vertical_walls or (
                row1 - 1,
                col,
            ) in self.p2_vertical_walls:
                return True

        return False

    def _has_path(self, start, target_row):
        visited = set()
        queue = deque([tuple(start)])

        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        while queue:
            row, col = queue.popleft()

            if row == target_row:
                return True

            for drow, dcol in directions:
                nrow, ncol = row + drow, col + dcol
                next_pos = (nrow, ncol)

                if not self.is_inside_grid(next_pos):
                    continue

                if next_pos in visited:
                    continue

                if self.is_blocked((row, col), next_pos):
                    continue

                visited.add(next_pos)
                queue.append(next_pos)

        return False

    def _players_have_path(self):
        p1_ok = self._has_path(self.p1_pos, target_row=0)
        p2_ok = self._has_path(self.p2_pos, target_row=self.grid_size - 1)
        return p1_ok and p2_ok
