import numpy as np
import pygame
import torch
from game_logic_Ai import GridGameAi, P1, P2

pygame.init()


class TrainingUI:
    """
    Visualizza le partite del training dei due agenti con pygame.
    Simile a pygame_ui.py ma ottimizzata per il training automatico.

    Uso:
        ui = TrainingUI(env, show_every=5, speed=30)

        # Durante il training:
        while not done:
            # ... logica di gioco ...
            ui.render()  # Visualizza il gioco
    """

    def __init__(self, env, show_every=1, speed=30, width_scale=1.0):
        """
        Args:
            env: istanza di GridGameAi
            show_every: visualizza ogni N partite (per velocizzare training)
            speed: tick rate del clock in FPS
            width_scale: scala della finestra (1.0 = 100% dello schermo)
        """
        self.env = env
        self.show_every = show_every
        self.speed = speed
        self.game_count = 0
        self.show_current = True  # Mostra sempre per default

        # Calcola dimensioni finestra
        self.CELL = 60
        self.SIDE_PANEL = 280

        self.W = self.CELL * env.grid_size + self.SIDE_PANEL
        self.H = self.CELL * env.grid_size

        # Applica scale
        self.W = int(self.W * width_scale)
        self.H = int(self.H * width_scale)
        self.CELL = int(self.CELL * width_scale)
        self.SIDE_PANEL = int(self.SIDE_PANEL * width_scale)

        # Crea finestra
        self.screen = pygame.display.set_mode((self.W, self.H))
        pygame.display.set_caption("Quoridor - Agent Training")
        self.clock = pygame.time.Clock()

        # Tracking stato
        self.step_count = 0
        self.winner = None
        self.last_action = None

    def new_game(self):
        """Chiamare questo all'inizio di ogni partita nel training."""
        self.game_count += 1
        # Show the first game immediately, then every N games after that.
        # e.g. with show_every=5 we show games 1, 6, 11, ... instead of waiting
        # until game 5 while the window stays black at startup.
        self.show_current = ((self.game_count - 1) % self.show_every) == 0
        self.step_count = 0
        self.winner = None
        self.last_action = None

    def render(self, game_num=None, step=None):
        """
        Disegna lo stato attuale del gioco.

        Args:
            game_num: numero della partita attuale (opzionale)
            step: numero dello step attuale (opzionale)
        """
        if not self.show_current:
            return  # Non disegnare se non è una partita da mostrare

        # Gestisci ctrl+c
        try:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    raise KeyboardInterrupt("UI chiuso")
        except:
            pass

        # Gestisci eventi pygame (permetti di chiudere la finestra)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                raise KeyboardInterrupt("Training visualization closed by user")

        self.screen.fill((255, 255, 255))

        # Disegna elementi di gioco
        self._draw_grid()
        self._draw_walls()
        self._draw_players()
        self._draw_sidebar(game_num, step)

        pygame.display.flip()
        self.clock.tick(self.speed)

    def _draw_grid(self):
        """Disegna la griglia del gioco."""
        for row in range(self.env.grid_size):
            for col in range(self.env.grid_size):
                screen_x = col * self.CELL
                screen_y = row * self.CELL
                rect = pygame.Rect(screen_x, screen_y, self.CELL, self.CELL)
                pygame.draw.rect(self.screen, (200, 200, 200), rect, 1)

    def _draw_players(self):
        """Disegna i due giocatori."""
        colors = {P1: (0, 0, 255), P2: (255, 0, 0)}

        for p, pos in [(P1, self.env.p1_pos), (P2, self.env.p2_pos)]:
            screen_x = pos[1] * self.CELL + self.CELL // 2
            screen_y = pos[0] * self.CELL + self.CELL // 2
            pygame.draw.circle(
                self.screen, colors[p], (screen_x, screen_y), self.CELL // 3
            )

            # Disegna numero del giocatore al centro del cerchio
            font = pygame.font.SysFont(None, int(self.CELL // 2))
            txt = font.render(str(p), True, (255, 255, 255))
            txt_rect = txt.get_rect(center=(screen_x, screen_y))
            self.screen.blit(txt, txt_rect)

    def _draw_walls(self):
        """Disegna i muri orizzontali e verticali."""
        # Muri orizzontali di p1
        for row, col in self.env.p1_horizontal_walls:
            pygame.draw.rect(
                self.screen,
                (0, 0, 200),
                pygame.Rect(
                    col * self.CELL,
                    (row + 1) * self.CELL - 5,
                    self.CELL * 2,
                    10,
                ),
            )

        # Muri verticali di p1
        for row, col in self.env.p1_vertical_walls:
            pygame.draw.rect(
                self.screen,
                (0, 0, 200),
                pygame.Rect(
                    (col + 1) * self.CELL - 5,
                    row * self.CELL,
                    10,
                    self.CELL * 2,
                ),
            )

        # Muri orizzontali di p2
        for row, col in self.env.p2_horizontal_walls:
            pygame.draw.rect(
                self.screen,
                (200, 0, 0),
                pygame.Rect(
                    col * self.CELL,
                    (row + 1) * self.CELL - 5,
                    self.CELL * 2,
                    10,
                ),
            )

        # Muri verticali di p2
        for row, col in self.env.p2_vertical_walls:
            pygame.draw.rect(
                self.screen,
                (200, 0, 0),
                pygame.Rect(
                    (col + 1) * self.CELL - 5,
                    row * self.CELL,
                    10,
                    self.CELL * 2,
                ),
            )

    def _draw_sidebar(self, game_num=None, step=None):
        """Disegna il pannello laterale con info."""
        panel_x = self.CELL * self.env.grid_size
        panel_width = self.SIDE_PANEL
        panel_height = self.H

        # Sfondo del pannello
        pygame.draw.rect(
            self.screen,
            (240, 240, 240),
            pygame.Rect(panel_x, 0, panel_width, panel_height),
        )

        font_small = pygame.font.SysFont(None, int(18 * (self.W / 720)))
        font_large = pygame.font.SysFont(None, int(24 * (self.W / 720)))
        font_title = pygame.font.SysFont(None, int(28 * (self.W / 720)))

        y_offset = 10
        line_height = int(28 * (self.W / 720))

        # Titolo
        txt_title = font_title.render("TRAINING", True, (0, 0, 0))
        self.screen.blit(txt_title, (panel_x + 10, y_offset))
        y_offset += line_height + 5

        # Numero partita
        if game_num is not None:
            txt = font_large.render(f"Game: {game_num}", True, (0, 0, 0))
            self.screen.blit(txt, (panel_x + 10, y_offset))
            y_offset += line_height

        # Step count
        if step is not None:
            txt = font_large.render(f"Step: {step}", True, (0, 0, 0))
            self.screen.blit(txt, (panel_x + 10, y_offset))
            y_offset += line_height

        y_offset += 10

        # Info giocatori
        txt_p1_title = font_large.render("Player 1 (Blue)", True, (0, 0, 200))
        self.screen.blit(txt_p1_title, (panel_x + 10, y_offset))
        y_offset += line_height

        txt_p1_pos = font_small.render(
            f"Pos: {int(self.env.p1_pos[0])}, {int(self.env.p1_pos[1])}",
            True,
            (0, 0, 200),
        )
        self.screen.blit(txt_p1_pos, (panel_x + 10, y_offset))
        y_offset += line_height - 5

        txt_p1_walls = font_small.render(
            f"Walls: {self.env.p1_available_walls}/10", True, (0, 0, 200)
        )
        self.screen.blit(txt_p1_walls, (panel_x + 10, y_offset))
        y_offset += line_height + 5

        txt_p2_title = font_large.render("Player 2 (Red)", True, (200, 0, 0))
        self.screen.blit(txt_p2_title, (panel_x + 10, y_offset))
        y_offset += line_height

        txt_p2_pos = font_small.render(
            f"Pos: {int(self.env.p2_pos[0])}, {int(self.env.p2_pos[1])}",
            True,
            (200, 0, 0),
        )
        self.screen.blit(txt_p2_pos, (panel_x + 10, y_offset))
        y_offset += line_height - 5

        txt_p2_walls = font_small.render(
            f"Walls: {self.env.p2_available_walls}/10", True, (200, 0, 0)
        )
        self.screen.blit(txt_p2_walls, (panel_x + 10, y_offset))
        y_offset += line_height + 5

        # Turn
        turn_player = self.env.turn
        turn_color = (0, 0, 200) if turn_player == P1 else (200, 0, 0)
        txt_turn = font_large.render(f"Turn: P{turn_player}", True, turn_color)
        self.screen.blit(txt_turn, (panel_x + 10, y_offset))
        y_offset += line_height + 10

        # Ultimo muro posizionato
        total_walls = (
            len(self.env.p1_horizontal_walls)
            + len(self.env.p1_vertical_walls)
            + len(self.env.p2_horizontal_walls)
            + len(self.env.p2_vertical_walls)
        )
        txt_walls = font_small.render(f"Total walls: {total_walls}/20", True, (0, 0, 0))
        self.screen.blit(txt_walls, (panel_x + 10, y_offset))

    def close(self):
        """Chiudi la finestra pygame."""
        pygame.quit()
