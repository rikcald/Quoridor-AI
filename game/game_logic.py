from collections import deque

P1 = 1
P2 = 2


class GridGame:
    def __init__(self, grid_size=9):
        self.grid_size = grid_size
        self.reset()

    def reset(self):
        self.grid = [[0] * self.grid_size for _ in range(self.grid_size)]

        self.p1_pos = [self.grid_size // 2, self.grid_size - 1]
        self.p2_pos = [self.grid_size // 2, 0]

        self.grid[self.p1_pos[0]][self.p1_pos[1]] = P1
        self.grid[self.p2_pos[0]][self.p2_pos[1]] = P2

        self.horizontal_walls = set()
        self.vertical_walls = set()

        self.p1_available_walls = 10
        self.p2_availablewalls = 10

        self.turn = P1

    def move(self, player, move):

        if player != self.turn:
            print("Non è il tuo turno")
            return

        moves = self.available_moves(player)

        if move not in moves:
            raise ValueError(f"Mossa non valida: {move}")

        action, direction = move

        if player == P1:
            current = self.p1_pos
        else:
            current = self.p2_pos

        # parsing direzione
        parts = direction.split("-")

        # movimento base
        base_dirs = {
            "up": (0, -1),
            "down": (0, 1),
            "left": (-1, 0),
            "right": (1, 0),
        }

        dx, dy = base_dirs[parts[0]]

        new_pos = [current[0] + dx, current[1] + dy]

        # caso jump
        if len(parts) == 2 and parts[1] == "jump":
            new_pos = [new_pos[0] + dx, new_pos[1] + dy]

        # caso diagonale
        elif len(parts) == 2:
            sdx, sdy = base_dirs[parts[1]]
            new_pos = [new_pos[0] + sdx, new_pos[1] + sdy]

        # aggiorna griglia
        self.grid[current[0]][current[1]] = 0
        self.grid[new_pos[0]][new_pos[1]] = player

        if player == P1:
            self.p1_pos = new_pos
        else:
            self.p2_pos = new_pos

        print(f"Player {player} moves {new_pos} via {direction}")
        self.turn = P2 if self.turn == P1 else P1

    def check_winner(self):

        if self.p1_pos[1] == 0:
            return P1
        elif self.p2_pos[1] == self.grid_size - 1:
            return P2
        return None

    def print_grid(self):
        print("-" * 30)

        for y in range(self.grid_size):
            # riga celle
            row = ""
            for x in range(self.grid_size):
                row += str(self.grid[x][y])

                # muro verticale
                if x < self.grid_size - 1:
                    if (x, y) in self.vertical_walls:
                        row += "|"
                    else:
                        row += " "
            print(row)

            # riga muri orizzontali
            if y < self.grid_size - 1:
                row = ""
                for x in range(self.grid_size):
                    if (x, y) in self.horizontal_walls:
                        row += "──"
                    else:
                        row += "  "
                print(row)

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
            "up": (0, -1),
            "down": (0, 1),
            "left": (-1, 0),
            "right": (1, 0),
        }

        for d, (dx, dy) in directions.items():
            adj = [current[0] + dx, current[1] + dy]

            if not self.is_inside_grid(adj):
                continue

            # caso normale: cella libera
            if (
                adj != opponent
                and self.grid[adj[0]][adj[1]] == 0
                and not self.is_blocked(current, adj)
            ):
                moves.append(("move", d))
                continue

            # caso salto
            if adj == opponent and not self.is_blocked(current, adj):
                jump = [adj[0] + dx, adj[1] + dy]

                if (
                    self.is_inside_grid(jump)
                    and self.grid[jump[0]][jump[1]] == 0
                    and not self.is_blocked(adj, jump)
                ):
                    moves.append(("move", f"{d}-jump"))
                else:
                    # diagonali
                    if d in ["up", "down"]:
                        sides = [("left", (-1, 0)), ("right", (1, 0))]
                    else:
                        sides = [("up", (0, -1)), ("down", (0, 1))]

                    for sd, (sdx, sdy) in sides:
                        diag = [adj[0] + sdx, adj[1] + sdy]

                        if (
                            self.is_inside_grid(diag)
                            and self.grid[diag[0]][diag[1]] == 0
                            and not self.is_blocked(adj, diag)
                        ):
                            moves.append(("move", f"{d}-{sd}"))

        return moves

    def is_inside_grid(self, pos):
        return 0 <= pos[0] < self.grid_size and 0 <= pos[1] < self.grid_size

    def place_wall(self, player, location, orientation):
        if player != self.turn:
            print("Non è il tuo turno")
            return False, None
        x, y = location

        if player == P1 and self.p1_available_walls <= 0:
            print("Non hai più muri disponibili")
            return False, "No walls available!"
        if player == P2 and self.p2_availablewalls <= 0:
            print("Non hai più muri disponibili")
            return False, "No walls available!"

        # limiti (muri stanno tra le celle)
        if x < 0 or y < 0 or x >= self.grid_size - 1 or y >= self.grid_size - 1:
            print("Posizione muro non valida (fuori range)")
            return False, None

        if orientation == "h":
            if (
                (x, y) in self.horizontal_walls
                or (x + 1, y) in self.horizontal_walls
                or (x - 1, y) in self.horizontal_walls
            ):
                return False, "Invalid placement!"

            if (x, y) in self.vertical_walls:
                return False, "Invalid placement!"

            self.horizontal_walls.add((x, y))

            if player == P1:
                self.p1_available_walls -= 1
            else:
                self.p2_availablewalls -= 1

        elif orientation == "v":
            if (
                (x, y) in self.vertical_walls
                or (x, y + 1) in self.vertical_walls
                or (x, y - 1) in self.vertical_walls
            ):
                return False, "Invalid placement!"

            if (x, y) in self.horizontal_walls:
                return False, "Invalid placement!"
            self.vertical_walls.add((x, y))
            if player == P1:
                self.p1_available_walls -= 1
            else:
                self.p2_availablewalls -= 1

        else:
            return False, None

        # CHECK PATH (after placing the wall, to allow rollback if it blocks all paths)
        if not self._players_have_path():
            print("Muro blocca completamente un giocatore!")

            # rollback
            if orientation == "h":
                self.horizontal_walls.remove((x, y))
                if player == P1:
                    self.p1_available_walls += 1
                else:
                    self.p2_availablewalls += 1
            else:
                self.vertical_walls.remove((x, y))
                if player == P1:
                    self.p1_available_walls += 1
                else:
                    self.p2_availablewalls += 1

            return False, "Invalid placement!"

        self.turn = P2 if self.turn == P1 else P1
        return True, None

    def is_blocked(self, a, b):
        x1, y1 = a
        x2, y2 = b

        # movimento verticale (su/giù)
        if x1 == x2:
            y = min(y1, y2)

            # muro orizzontale blocca
            if (x1, y) in self.horizontal_walls:
                return True
            if (x1 - 1, y) in self.horizontal_walls:
                return True

        # movimento orizzontale (sx/dx)
        elif y1 == y2:
            x = min(x1, x2)

            # muro verticale blocca
            if (x, y1) in self.vertical_walls:
                return True
            if (x, y1 - 1) in self.vertical_walls:
                return True

        return False

    def _has_path(self, start, target_row):
        visited = set()
        queue = deque([tuple(start)])

        directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]

        while queue:
            x, y = queue.popleft()

            # 🎯 raggiunta la goal
            if y == target_row:
                return True

            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                next_pos = (nx, ny)

                if not self.is_inside_grid([nx, ny]):
                    continue

                if next_pos in visited:
                    continue

                # muro blocca?
                if self.is_blocked((x, y), next_pos):
                    continue

                visited.add(next_pos)
                queue.append(next_pos)

        return False

    def _players_have_path(self):
        p1_ok = self._has_path(self.p1_pos, target_row=0)
        p2_ok = self._has_path(self.p2_pos, target_row=self.grid_size - 1)
        return p1_ok and p2_ok
