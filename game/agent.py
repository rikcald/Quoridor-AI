from collections import deque

import torch
import random
import numpy as np
from game_logic_Ai import P1, P2, TOTAL_ACTIONS, GridGameAi
from model import Linear_QNet, QTrainer

MAX_MEMORY = 100000
BATCH_SIZE = 1000
LR = 0.001


class Agent:
    def __init__(self, player_id):
        self.player_id = player_id
        self.n_games = 0
        self.epsilon = 0  # For exploration
        self.gamma = 0.9  # Discount rate
        self.episode_reward = 0  # Reward SOLO per questo agente

        # if memory exceeds max, the oldest experience is removed (popleft)
        self.memory = deque(maxlen=MAX_MEMORY)

        self.model = Linear_QNet(567, 256, TOTAL_ACTIONS)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    # Choose action based on environment's action mask
    def get_action(self, state, env):
        """
        L'agente sceglie un'azione per il suo player.
        Non è necessario passare il player poiché l'agente conosce il suo player_id.
        """

        # exploitation vs exploration tradeoff
        # decrese epsilon (aka exploration) as the number of games increases
        self.epsilon = 80 - self.n_games
        if random.randint(0, 200) < self.epsilon:
            # print("random action")
            # get action mask for the current player (1 for valid, 0 for invalid) e.g. [1, 0, 1, 1, 0, 1] where indices correspond to actions and values indicate validity
            mask = env.get_action_mask(self.player_id)

            # get valid action indices e.g. [0, 2, 5]
            valid_actions = np.where(mask == 1)[0]

            if len(valid_actions) == 0:
                raise Exception(
                    f"No valid actions available for player {self.player_id}"
                )

            return int(np.random.choice(valid_actions))

        else:
            # print("predicted action")
            # Convert state to tensor and flatten it
            state_tensor = torch.tensor(state, dtype=torch.float).flatten()
            # Add batch dimension (1, 144)
            prediction = self.model(state_tensor.unsqueeze(0))

            # Get valid action mask
            mask = env.get_action_mask(self.player_id)
            mask_tensor = torch.tensor(mask, dtype=torch.float)

            # Apply mask to predictions: set invalid actions to very negative values
            masked_prediction = prediction.clone()
            masked_prediction[0, mask_tensor == 0] = -1e9  # -1e9 for invalid actions
            # Get the best valid action
            move = torch.argmax(masked_prediction).item()

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


def train_agents(env, agent1, agent2, num_games=1000):
    plot_scores_p1 = []
    plot_scores_p2 = []
    record_p1 = 0
    record_p2 = 0

    # Check if agents are controlling the correct players
    if agent1.player_id == agent2.player_id:
        raise ValueError("Both agents cannot control the same player!")

    for game_num in range(num_games):
        env.reset()
        agent1.episode_reward = 0
        agent2.episode_reward = 0
        done = False
        step_count = 0

        # DEBUG: traccia i tipi di azione scelti
        move_count = 0
        wall_count = 0
        invalid_count = 0

        while not done and step_count < 2500:  # Max steps per evitare loop infiniti
            # Determina chi deve giocare basato su game.turn
            current_player = env.turn
            state_old = env.get_state()

            # Chiedi l'azione all'agente appropriato
            if current_player == P1:
                action = agent1.get_action(state_old, env)
            else:
                action = agent2.get_action(state_old, env)

            # Esegui l'azione nel gioco
            state_new, reward, done, info = env.step(action)

            # DEBUG: classifica l'azione
            if info.get("invalid", False):
                invalid_count += 1
            elif action < 16:  # NUM_MOVE_ACTIONS
                move_count += 1
            else:
                wall_count += 1

            # Salva l'esperienza e il reward SOLO per l'agente che ha giocato
            if current_player == P1:
                agent1.train_short_memory(state_old, action, reward, state_new, done)
                agent1.remember(state_old, action, reward, state_new, done)
                agent1.episode_reward += reward
            else:
                agent2.train_short_memory(state_old, action, reward, state_new, done)
                agent2.remember(state_old, action, reward, state_new, done)
                agent2.episode_reward += reward

            step_count += 1
            # game.print_grid()

        # End of game - allenamento long memory per entrambi
        agent1.n_games += 1
        agent2.n_games += 1
        agent1.train_long_memory()
        agent2.train_long_memory()

        # Salva i modelli se raggiungono il record
        final_score_p1 = agent1.episode_reward
        final_score_p2 = agent2.episode_reward

        if final_score_p1 > record_p1:
            record_p1 = final_score_p1
            agent1.model.save("Linear_QNet_model_P1.pth")

        if final_score_p2 > record_p2:
            record_p2 = final_score_p2
            agent2.model.save("Linear_QNet_model_P2.pth")

        plot_scores_p1.append(final_score_p1)
        plot_scores_p2.append(final_score_p2)

        if agent1.n_games % 10 == 0:
            print(
                f"\nGame {agent1.n_games} | "
                f"P1 Score: {final_score_p1:.2f} (Record: {record_p1:.2f}) | "
                f"P2 Score: {final_score_p2:.2f} (Record: {record_p2:.2f}) | "
                f"Steps: {step_count} | "
                f"Moves: {move_count}, Walls: {wall_count}, Invalid: {invalid_count}"
            )
            # plot(plot_scores_p1)  # Opzionale: mostra i plot


# Test code

if __name__ == "__main__":
    # Training: due agenti separati si affrontano
    print("\n=== Starting P1 vs P2 Agent Training ===")
    agent1 = Agent(P1)
    agent2 = Agent(P2)
    env = GridGameAi()
    train_agents(env, agent1, agent2, num_games=10)

    # Test: agenti casuali
    # print("\n=== Testing Random vs Random ===")
    # agent = Agent(P1)
    # agent.test_random_vs_random(10)


# TODO problema del piazzamento invalido risolto, ora il problema è che gli agenti non imparano in maniera efficace, prima di tutto io staccherei la funzione statica train_vs_agent
# TODO dalla clase agent e la farei fuori, in modo tale che richieda env,agente1,agente2 e faccia le sue cose, magari cercare un modo per far condividere il modello agli agenti, non so perchè ma il fatto che siano due modelli mi puzza (oppure è una cosa buona perchè posso vedere magari CNN vs NN per esempio piu in la)
