from collections import deque

import numpy as np

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
NUM_MOVE_ACTIONS = len(MOVE_ACTIONS)

# Board configuration defaults for the fast training variant.
# To train on the classic board, use grid_size=9 and max_walls=10 when
# creating GridGameAi and AlphaZeroSelfPlayAgent.
DEFAULT_GRID_SIZE = 5
DEFAULT_MAX_WALLS = 4


def num_wall_positions_for_grid(grid_size):
    return (grid_size - 1) ** 2


def total_actions_for_grid(grid_size):
    return NUM_MOVE_ACTIONS + 2 * num_wall_positions_for_grid(grid_size)


NUM_WALL_POS = num_wall_positions_for_grid(DEFAULT_GRID_SIZE)
TOTAL_ACTIONS = total_actions_for_grid(DEFAULT_GRID_SIZE)
BASE_DIRECTIONS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
PATH_DIRECTIONS = tuple(BASE_DIRECTIONS.values())


def flip_direction_for_p2(direction):
    """
    P2 is normalized with a vertical flip only.

    e.g.
    - "up" -> "down"
    - "down-jump" -> "up-jump"
    - "left" stays "left"
    """
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
    """
    Quoridor game state used by the AlphaZero / MCTS pipeline.

    Main ideas:
    - the object stores the real board
    - the network sees canonical player-centric states
    - MCTS clones this object to explore hypothetical continuations
    """

    def __init__(self, grid_size=DEFAULT_GRID_SIZE, max_walls=DEFAULT_MAX_WALLS):
        self.grid_size = grid_size
        self.max_walls = max_walls
        self.num_wall_positions = num_wall_positions_for_grid(grid_size)
        self.total_actions = total_actions_for_grid(grid_size)
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

        self.p1_available_walls = self.max_walls
        self.p2_available_walls = self.max_walls
        self.turn = P1
        self._valid_actions_cache = {}

    def clone(self):
        cloned = self.__class__.__new__(self.__class__)
        cloned.grid_size = self.grid_size
        cloned.max_walls = self.max_walls
        cloned.num_wall_positions = self.num_wall_positions
        cloned.total_actions = self.total_actions
        cloned.grid = self.grid.copy()
        cloned.p1_pos = self.p1_pos.copy()
        cloned.p2_pos = self.p2_pos.copy()
        cloned.p1_horizontal_walls = set(self.p1_horizontal_walls)
        cloned.p1_vertical_walls = set(self.p1_vertical_walls)
        cloned.p2_horizontal_walls = set(self.p2_horizontal_walls)
        cloned.p2_vertical_walls = set(self.p2_vertical_walls)
        cloned.p1_available_walls = self.p1_available_walls
        cloned.p2_available_walls = self.p2_available_walls
        cloned.turn = self.turn
        cloned._valid_actions_cache = {
            player: actions.copy()
            for player, actions in self._valid_actions_cache.items()
        }
        return cloned

    def _invalidate_action_cache(self):
        # Legal actions depend on pawn positions, walls, wall counts, and turn.
        # Any real state mutation clears cached actions for both players.
        self._valid_actions_cache.clear()

    def get_current_player(self):
        return self.turn

    def _flip_wall_row_for_p2(self, row):
        return (self.grid_size - 2) - row

    def _transform_wall_row(self, row, player):
        if player == P1:
            return row
        return self._flip_wall_row_for_p2(row)

    def decode_action(self, action, player=None):
        if player is None:
            player = self.turn

        if action < NUM_MOVE_ACTIONS:
            direction = MOVE_ACTIONS[action]
            if player == P2:
                direction = flip_direction_for_p2(direction)
            return "move", direction

        if action < NUM_MOVE_ACTIONS + self.num_wall_positions:
            wall_index = action - NUM_MOVE_ACTIONS
            row = wall_index // (self.grid_size - 1)
            col = wall_index % (self.grid_size - 1)
            if player == P2:
                row = self._flip_wall_row_for_p2(row)
            return "wall", (row, col), "h"

        wall_index = action - NUM_MOVE_ACTIONS - self.num_wall_positions
        row = wall_index // (self.grid_size - 1)
        col = wall_index % (self.grid_size - 1)
        if player == P2:
            row = self._flip_wall_row_for_p2(row)
        return "wall", (row, col), "v"

    def encode_action(self, action_type, *args, player=None):
        if player is None:
            player = self.turn

        if action_type == "move":
            direction = args[0]
            if player == P2:
                direction = flip_direction_for_p2(direction)
            return MOVE_ACTIONS.index(direction)

        if action_type == "wall":
            row, col = args[0]
            orientation = args[1]
            if player == P2:
                row = self._flip_wall_row_for_p2(row)

            wall_index = row * (self.grid_size - 1) + col
            if orientation == "h":
                return NUM_MOVE_ACTIONS + wall_index
            return NUM_MOVE_ACTIONS + self.num_wall_positions + wall_index

        raise ValueError(f"Invalid action encoding: {action_type}, {args}")

    def get_state_player_centric(self, player):
        """
        Canonical network input with 6 planes:
        0 = current player
        1 = opponent
        2 = horizontal walls
        3 = vertical walls
        4 = my remaining walls
        5 = opponent remaining walls
        """
        size = self.grid_size
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
            row_transform = lambda row: size - 1 - row

        state[0, row_transform(my_pos[0]), my_pos[1]] = 1.0
        state[1, row_transform(opp_pos[0]), opp_pos[1]] = 1.0

        for r, c in my_horizontal_walls:
            state[2, self._transform_wall_row(r, player), c] = 1.0
        for r, c in opp_horizontal_walls:
            state[2, self._transform_wall_row(r, player), c] = 1.0

        for r, c in my_vertical_walls:
            state[3, self._transform_wall_row(r, player), c] = 1.0
        for r, c in opp_vertical_walls:
            state[3, self._transform_wall_row(r, player), c] = 1.0

        wall_scale = max(1, self.max_walls)
        state[4, :, :] = my_available_walls / wall_scale
        state[5, :, :] = opp_available_walls / wall_scale

        return state

    def get_canonical_state(self, player=None):
        if player is None:
            player = self.turn
        return self.get_state_player_centric(player)

    def get_state(self):
        return self.get_canonical_state(self.turn)

    def get_action_mask(self, player):
        mask = np.zeros(self.total_actions, dtype=np.float32)
        mask[self.get_valid_actions(player)] = 1.0
        return mask

    def get_valid_actions(self, player=None):
        if player is None:
            player = self.turn

        cached_actions = self._valid_actions_cache.get(player)
        if cached_actions is not None:
            return cached_actions.copy()

        actions = []
        for move in self.available_moves(player):
            actions.append(self.encode_action(*move, player=player))

        for row in range(self.grid_size - 1):
            for col in range(self.grid_size - 1):
                if self._is_valid_wall(player, (row, col), "h"):
                    actions.append(
                        self.encode_action("wall", (row, col), "h", player=player)
                    )
                if self._is_valid_wall(player, (row, col), "v"):
                    actions.append(
                        self.encode_action("wall", (row, col), "v", player=player)
                    )

        valid_actions = np.array(actions, dtype=np.int64)
        self._valid_actions_cache[player] = valid_actions
        return valid_actions.copy()

    def is_terminal(self):
        return self.check_winner() is not None

    def get_winner(self):
        return self.check_winner()

    def get_outcome_for_player(self, player):
        winner = self.check_winner()
        if winner is None:
            return 0
        return 1 if winner == player else -1

    def shortest_path_distance_to_goal(self, player):
        """
        Return the fewest pawn moves needed for `player` to reach its goal row.

        This is used only as a soft timeout adjudication signal: when self-play
        hits the move limit, the player with the shorter legal path is treated
        as slightly ahead instead of labeling the whole unfinished game a draw.
        """
        if player == P1:
            start = self.p1_pos
            target_row = 0
        else:
            start = self.p2_pos
            target_row = self.grid_size - 1

        start_pos = (int(start[0]), int(start[1]))
        visited = {start_pos}
        queue = deque([(start_pos, 0)])

        while queue:
            (row, col), distance = queue.popleft()
            if row == target_row:
                return distance

            for drow, dcol in PATH_DIRECTIONS:
                nrow, ncol = row + drow, col + dcol
                next_pos = (nrow, ncol)
                if (
                    nrow < 0
                    or nrow >= self.grid_size
                    or ncol < 0
                    or ncol >= self.grid_size
                ):
                    continue
                if next_pos in visited:
                    continue
                if self.is_blocked((row, col), next_pos):
                    continue

                visited.add(next_pos)
                queue.append((next_pos, distance + 1))

        return float("inf")

    def get_timeout_adjudication_winner(self):
        """
        Pick a soft winner for unfinished games by shortest legal path length.

        Equal distances remain a real draw, because neither side has a clear
        distance advantage at the truncation point.
        """
        p1_distance = self.shortest_path_distance_to_goal(P1)
        p2_distance = self.shortest_path_distance_to_goal(P2)

        if p1_distance < p2_distance:
            return P1
        if p2_distance < p1_distance:
            return P2
        return None

    def step(self, action):
        """
        Apply one canonical action index.

        Output:
        - next canonical state for the new player to move
        - done
        - info
        """
        player = self.turn
        decoded = self.decode_action(action, player=player)

        if decoded[0] == "move":
            _, direction = decoded
            try:
                self.move(player, ("move", direction))
            except ValueError:
                return self.get_state(), False, {"invalid": True, "winner": None}
        else:
            _, location, orientation = decoded
            success, _ = self.place_wall(player, location, orientation)
            if not success:
                return self.get_state(), False, {"invalid": True, "winner": None}

        winner = self.check_winner()
        done = winner is not None
        return self.get_state(), done, {"invalid": False, "winner": winner}

    def apply_action(self, action):
        return self.step(action)

    def next_state(self, action):
        cloned_state = self.clone()
        cloned_state.apply_action(action)
        return cloned_state

    def next_state_from_valid_action(self, action):
        cloned_state = self.clone()
        cloned_state._apply_valid_action_without_rechecking(action)
        return cloned_state

    def _apply_valid_action_without_rechecking(self, action):
        action_type, *decoded_args = self.decode_action(action, player=self.turn)

        if action_type == "move":
            self._move_without_rechecking(self.turn, decoded_args[0])
        else:
            location, orientation = decoded_args
            self._place_wall_without_rechecking(self.turn, location, orientation)

    def to_dict(self):
        return {
            "grid_size": self.grid_size,
            "max_walls": self.max_walls,
            "num_wall_positions": self.num_wall_positions,
            "total_actions": self.total_actions,
            "turn": self.turn,
            "p1_pos": tuple(self.p1_pos.tolist()),
            "p2_pos": tuple(self.p2_pos.tolist()),
            "p1_horizontal_walls": sorted(self.p1_horizontal_walls),
            "p1_vertical_walls": sorted(self.p1_vertical_walls),
            "p2_horizontal_walls": sorted(self.p2_horizontal_walls),
            "p2_vertical_walls": sorted(self.p2_vertical_walls),
            "p1_available_walls": self.p1_available_walls,
            "p2_available_walls": self.p2_available_walls,
        }

    def string_representation(self):
        return str(self.to_dict())

    def move(self, player, move):
        if player != self.turn:
            return

        moves = self.available_moves(player)
        if move not in moves:
            raise ValueError(f"Mossa non valida: {move}")

        _, direction = move
        current = self.p1_pos.copy() if player == P1 else self.p2_pos.copy()
        parts = direction.split("-")

        drow, dcol = BASE_DIRECTIONS[parts[0]]
        new_pos = current + np.array([drow, dcol])

        if len(parts) == 2 and parts[1] == "jump":
            new_pos = new_pos + np.array([drow, dcol])
        elif len(parts) == 2:
            sdrow, sdcol = BASE_DIRECTIONS[parts[1]]
            new_pos = new_pos + np.array([sdrow, sdcol])

        self._set_player_position(player, current, new_pos)
        return direction

    def _move_without_rechecking(self, player, direction):
        current = self.p1_pos.copy() if player == P1 else self.p2_pos.copy()
        parts = direction.split("-")
        drow, dcol = BASE_DIRECTIONS[parts[0]]
        new_pos = current + np.array([drow, dcol])

        if len(parts) == 2 and parts[1] == "jump":
            new_pos = new_pos + np.array([drow, dcol])
        elif len(parts) == 2:
            sdrow, sdcol = BASE_DIRECTIONS[parts[1]]
            new_pos = new_pos + np.array([sdrow, sdcol])

        self._set_player_position(player, current, new_pos)

    def _set_player_position(self, player, current, new_pos):
        self.grid[current[0], current[1]] = 0
        self.grid[new_pos[0], new_pos[1]] = player

        if player == P1:
            self.p1_pos = new_pos
        else:
            self.p2_pos = new_pos

        self.turn = P2 if self.turn == P1 else P1
        self._invalidate_action_cache()

    def check_winner(self):
        if self.p1_pos[0] == 0:
            return P1
        if self.p2_pos[0] == self.grid_size - 1:
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
        for d, (drow, dcol) in BASE_DIRECTIONS.items():
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
        row, col = pos
        return 0 <= row < self.grid_size and 0 <= col < self.grid_size

    def place_wall(self, player, location, orientation):
        if player != self.turn:
            return False, None

        row, col = location
        if not self._is_valid_wall(player, location, orientation):
            return False, "Invalid placement!"

        self._place_wall_without_rechecking(player, location, orientation)
        return True, None

    def _place_wall_without_rechecking(self, player, location, orientation):
        row, col = location

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
        self._invalidate_action_cache()

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
        start_pos = (int(start[0]), int(start[1]))
        visited = {start_pos}
        queue = deque([start_pos])
        grid_size = self.grid_size

        while queue:
            row, col = queue.popleft()
            if row == target_row:
                return True

            for drow, dcol in PATH_DIRECTIONS:
                nrow, ncol = row + drow, col + dcol
                next_pos = (nrow, ncol)
                if nrow < 0 or nrow >= grid_size or ncol < 0 or ncol >= grid_size:
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
