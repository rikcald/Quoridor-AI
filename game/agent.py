from collections import deque

import torch
import random
import numpy as np
from game_logic_Ai import P1, P2, GridGameAi

MAX_MEMORY = 100000
BATCH_SIZE = 1000
LR = 0.001


class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0  # For exploration
        self.gamma = 0  # Discount rate

        # if memory exceeds max, the oldest experience is removed (popleft)
        self.memory = deque(maxlen=MAX_MEMORY)

        self.model = None  # TODO
        self.trainer = None  # TODO

    # Choose action based on environment's action mask
    def get_action(self, state, env, player):

        # exploitation vs exploration tradeoff
        # decrese epsilon (aka exploration) as the number of games increases
        self.epsilon = 80 - self.n_games
        if random.randint(0, 200) < self.epsilon:
            # get action mask for the current player (1 for valid, 0 for invalid) e.g. [1, 0, 1, 1, 0, 1] where indices correspond to actions and values indicate validity
            mask = env.get_action_mask(player)

            # get valid action indices e.g. [0, 2, 5]
            valid_actions = np.where(mask == 1)[0]

            if len(valid_actions) == 0:
                raise Exception("No valid actions available")

            return np.random.choice(valid_actions)
        else:
            prediction = self.model(state)
            move = torch.argmax(prediction).item()

            return move

    def remember(self, state, action, reward, next_state, done):
        # store experience in memory
        self.memory.append((state, action, reward, next_state, done))
        self.episode_reward += reward

    def train_short_memory(self, state, action, reward, next_state, done):
        # Train the model on a single experience (state, action, reward, next_state, done)
        pass

    def train_long_memory(self):
        # Train the model on a batch of experiences from memory

        pass

    def train(self):
        plot_scores = []
        plot_mean_scores = []
        total_score = 0
        record = 0  # record del punteggio più alto raggiunto finora
        agent = Agent()
        game = GridGameAi()
        while True:
            # get old state
            state_old = game.get_state()  # TODO: forse serve convertire lo stato del gioco in un formato adatto al modello (es. array numpy)

            # get move
            final_move = agent.get_action(state_old)

            # perform move and get new state
            reward, done, score = game.step(final_move)
            state_new = game.get_state()

            # train short memory
            agent.train_short_memory(state_old, final_move, reward, state_new, done)

            # remember
            agent.remember(state_old, final_move, reward, state_new, done)

            if done:
                # train long memory, plot result
                game.reset()
                agent.n_games += 1
                agent.train_long_memory()

                if score > record:
                    record = score
                    # agent.model.save() (salva il modello se raggiunge un nuovo record)

                print("Game", agent.n_games, "Score", score, "Record:", record)

    def random_agent_action(self, env, player):
        # get action mask for the current player (1 for valid, 0 for invalid) e.g. [1, 0, 1, 1, 0, 1] where indices correspond to actions and values indicate validity
        mask = env.get_action_mask(player)

        # get valid action indices e.g. [0, 2, 5]
        valid_actions = np.where(mask == 1)[0]

        if len(valid_actions) == 0:
            raise Exception("No valid actions available")

        return np.random.choice(valid_actions)

    def play_random_game(self, env, max_steps=10000, render=False):
        state = env.reset()
        done = False
        step_count = 0
        p1_episode_reward = 0
        p2_episode_reward = 0

        while not done and step_count < max_steps:
            player = env.turn

            action = self.random_agent_action(env, player)

            next_state, reward, done, info = env.step(action)
            if player == P1:
                p1_episode_reward += reward
            else:
                p2_episode_reward += reward
            if render:
                print(
                    f"\nStep {step_count} - Player {player} - p1 Reward: {p1_episode_reward}, p2 Reward: {p2_episode_reward}"
                )
                print(f"Action: {action} -> {env.decode_action(action)}")
                # env.print_grid()

            state = next_state
            step_count += 1

        winner = env.check_winner()

        return {
            "winner": winner,
            "steps": step_count,
            "done": done,
            "p1_reward": p1_episode_reward,
            "p2_reward": p2_episode_reward,
        }

    def test_random_vs_random(self, num_games=50):
        env = GridGameAi()

        results = {"P1": 0, "P2": 0, "draw": 0, "invalid_games": 0}

        for i in range(num_games):
            game = self.play_random_game(env)

            if not game["done"]:
                results["draw"] += 1
                results["invalid_games"] += 1
            else:
                if game["winner"] == P1:
                    results["P1"] += 1
                elif game["winner"] == P2:
                    results["P2"] += 1

        print("\n=== RESULTS ===")
        print(results)


# Test code

agent = Agent()
env = GridGameAi()

# debug visivo (1 partita)
agent.play_random_game(env, render=True)

# test robustezza
# agent.test_random_vs_random(10)
