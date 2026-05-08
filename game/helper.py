import matplotlib.pyplot as plt


class LivePlotter:
    def __init__(self):
        plt.ion()  # interactive mode ON

        self.fig, self.axs = plt.subplots(2, 3, figsize=(15, 8))

        self.fig.suptitle("Training Stats")

    def update(self, stats):
        # clear axes
        for ax in self.axs.flat:
            ax.clear()

        # --- Steps ---
        self.axs[0, 0].plot(stats["steps"])
        self.axs[0, 0].set_title("Steps per Game")

        # --- Rewards ---
        self.axs[0, 1].plot(stats["p1_rewards"], label="P1")
        self.axs[0, 1].plot(stats["p2_rewards"], label="P2")
        self.axs[0, 1].set_title("Rewards")
        self.axs[0, 1].legend()

        # --- Walls ---
        self.axs[1, 0].plot(stats["walls"])
        self.axs[1, 0].set_title("Walls per Game")

        # --- Wins (cumulative, più utile) ---
        p1_wins = [sum(stats["p1_wins"][: i + 1]) for i in range(len(stats["p1_wins"]))]
        p2_wins = [sum(stats["p2_wins"][: i + 1]) for i in range(len(stats["p2_wins"]))]

        self.axs[1, 1].plot(p1_wins, label="P1 wins")
        self.axs[1, 1].plot(p2_wins, label="P2 wins")
        self.axs[1, 1].set_title("Cumulative Wins")
        self.axs[1, 1].legend()

        # --- Exploration Rate ---
        if "exploration_rate_p1" in stats and "exploration_rate_p2" in stats:
            self.axs[1, 2].plot(stats["exploration_rate_p1"], label="P1 Epsilon")
            self.axs[1, 2].plot(stats["exploration_rate_p2"], label="P2 Epsilon")
            self.axs[1, 2].set_title("Exploration Rate (Epsilon)")
            self.axs[1, 2].legend()
        else:
            self.axs[1, 2].text(0.5, 0.5, "No exploration data yet", ha="center", va="center")
            self.axs[1, 2].set_title("Exploration Rate (Epsilon)")

        # IMPORTANTISSIMO: non bloccare + refresh UI
        self.fig.tight_layout()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.01)
