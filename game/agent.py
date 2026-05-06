from collections import deque

import torch
import random
import numpy as np
from game_logic_Ai import P1, P2, TOTAL_ACTIONS, GridGameAi
from model import Linear_QNet, QTrainer
from helper import plot

MAX_MEMORY = 100000
BATCH_SIZE = 1000
LR = 0.001


class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0  # For exploration
        self.gamma = 0.9  # Discount rate
        self.episode_reward = 0  # Reward accumulation for current episode

        # if memory exceeds max, the oldest experience is removed (popleft)
        self.memory = deque(maxlen=MAX_MEMORY)

        self.model = Linear_QNet(567, 256, TOTAL_ACTIONS)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    # Choose action based on environment's action mask
    def get_action(self, state, env, player):

        # exploitation vs exploration tradeoff
        # decrese epsilon (aka exploration) as the number of games increases
        self.epsilon = 80 - self.n_games
        if random.randint(0, 200) < self.epsilon:
            # get action mask for the current player (1 for valid, 0 for invalid) e.g. [1, 0, 1, 1, 0, 1] where indices correspond to actions and values indicate validity
            mask = env.get_action_mask(player)
            print("a")
            # get valid action indices e.g. [0, 2, 5]
            valid_actions = np.where(mask == 1)[0]

            if len(valid_actions) == 0:
                raise Exception("No valid actions available")
            # print(valid_actions)
            return np.random.choice(valid_actions)

        else:
            print("b")
            # Convert state to tensor and flatten it
            state_tensor = torch.tensor(state, dtype=torch.float).flatten()
            prediction = self.model(
                state_tensor.unsqueeze(0)
            )  # Add batch dimension (1, 144)

            # Get valid action mask
            mask = env.get_action_mask(player)
            mask_tensor = torch.tensor(mask, dtype=torch.float)

            # Apply mask to predictions: set invalid actions to very negative values
            masked_prediction = prediction.clone()
            masked_prediction[0, mask_tensor == 0] = -1e9  # -1e9 for invalid actions

            # Get the best valid action
            move = torch.argmax(masked_prediction).item()
            print(move)
            print(prediction)
            print(masked_prediction)
            return move

    def remember(self, state, action, reward, next_state, done):
        # store experience in memory
        self.memory.append((state, action, reward, next_state, done))

    def train_short_memory(self, state, action, reward, next_state, done):
        # Train the model on a single experience (state, action, reward, next_state, done)
        self.trainer.train_step(state, action, reward, next_state, done)

    def train_long_memory(self):
        # Train the model on a batch of experiences from memory
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)  # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)

        # Convert to numpy arrays and stacks for batch training
        states = np.array(states)  # (batch_size, 7, 9, 9)
        actions = np.array(actions)  # (batch_size,)
        rewards = np.array(rewards)  # (batch_size,)
        next_states = np.array(next_states)  # (batch_size, 7, 9, 9)
        dones = np.array(dones)  # (batch_size,)

        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_vs_agent(self, num_games=1000):
        """
        Metodo di training dove due agenti si affrontano.
        P1 e P2 si alternano le mosse.
        """
        plot_scores = []
        record = 0

        # Entrambi gli agenti usano lo stesso modello per semplicità
        # (in una versione più avanzata potrebbero avere modelli separati)
        agent = Agent()

        for game_num in range(num_games):
            game = GridGameAi()
            agent.episode_reward = 0
            done = False
            step_count = 0

            while not done and step_count < 500:  # Max steps per evitare loop infiniti
                # Current player fa la mossa
                current_player = game.turn
                state_old = game.get_state()

                # Get action from agent
                action = agent.get_action(state_old, game, current_player)
                # print(action)
                # Execute action
                state_new, reward, done, info = game.step(action)

                # Train short memory
                agent.train_short_memory(state_old, action, reward, state_new, done)

                # Remember experience
                agent.remember(state_old, action, reward, state_new, done)
                agent.episode_reward += reward

                step_count += 1

            # End of game
            agent.n_games += 1
            agent.train_long_memory()

            final_score = agent.episode_reward

            if final_score > record:
                record = final_score
                agent.model.save()

            print(
                f"Game {agent.n_games} | Score: {final_score:.2f} | Record: {record:.2f} | Steps: {step_count}"
            )
            plot_scores.append(final_score)

            if agent.n_games % 10 == 0:
                plot(plot_scores)

    def train(self):
        # Deprecated: use train_vs_agent instead
        self.train_vs_agent()

    def random_agent_action(self, env, player):
        # get action mask for the current player (1 for valid, 0 for invalid) e.g. [1, 0, 1, 1, 0, 1] where indices correspond to actions and values indicate validity
        mask = env.get_action_mask(player)

        # get valid action indices e.g. [0, 2, 5]
        valid_actions = np.where(mask == 1)[0]

        if len(valid_actions) == 0:
            raise Exception("No valid actions available")

        return np.random.choice(valid_actions)

    def play_random_game(self, env, max_steps=10000, render=False):
        env.reset()
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

        print("\n=== RANDOM vs RANDOM RESULTS ===")
        print(f"P1 wins: {results['P1']}")
        print(f"P2 wins: {results['P2']}")
        print(f"Draws: {results['draw']}")


# Test code

if __name__ == "__main__":
    agent = Agent()

    # Training: due agenti si affrontano
    print("\n=== Starting Agent vs Agent Training ===")
    agent.train_vs_agent(num_games=2)

    # Test: agenti casuali
    # print("\n=== Testing Random vs Random ===")
    # agent.test_random_vs_random(10)

    # debug visivo (1 partita)
    # env = GridGameAi()
    # agent.play_random_game(env, render=True)

# TODO fix: nel get_action quando si sceglie un'azione random, sono disponibili anche quelle azioni (come up-jump) che dovrebbero essere illegali in quel momento.
# TODO Questo perché il mask restituito da get_action_mask è errato (non tiene conto di tutte le regole del gioco). Di conseguenza, l'agente potrebbe scegliere un'azione che sembra valida secondo il mask, ma che in realtà non lo è. Per risolvere questo problema, è necessario correggere la logica all'interno di get_action_mask per assicurarsi che rifletta accuratamente tutte le regole del gioco e le condizioni attuali del board.

# TODO fix: le mask sono rotte, visto che una volta piazzato il muro, printando la masked_prediction non ci sono valori negativi estremamente grandi,
# TODO come se il mask non venisse applicato correttamente. Di conseguenza, l'agente potrebbe scegliere un'azione che sembra valida secondo il mask, ma che in realtà non lo è. Per risolvere questo problema, è necessario correggere la logica all'interno di get_action_mask per assicurarsi che rifletta accuratamente tutte le regole del gioco e le condizioni attuali del board.
