import torch
import random
import numpy as np
from game_logic_Ai import P1, P2, GridGameAi

MAX_MEMORY = 10000
BATCH_SIZE = 1000
LR = 0.001
P1_REWARD = 0
P2_REWARD = 0


def random_agent_action(env, player):
    # get action mask for the current player (1 for valid, 0 for invalid) e.g. [1, 0, 1, 1, 0, 1] where indices correspond to actions and values indicate validity
    mask = env.get_action_mask(player)

    # get valid action indices e.g. [0, 2, 5]
    valid_actions = np.where(mask == 1)[0]

    if len(valid_actions) == 0:
        raise Exception("No valid actions available")

    return np.random.choice(valid_actions)


def play_random_game(env, max_steps=10000, render=False):
    state = env.reset()
    done = False
    step_count = 0

    while not done and step_count < max_steps:
        player = env.turn

        action = random_agent_action(env, player)

        next_state, reward, done, info = env.step(action)
        if player == P1:
            global P1_REWARD
            P1_REWARD += reward
        else:
            global P2_REWARD
            P2_REWARD += reward
        if render:
            print(
                f"\nStep {step_count} - Player {player} - p1 Reward: {P1_REWARD}, p2 Reward: {P2_REWARD}"
            )
            print(f"Action: {action} -> {env.decode_action(action)}")
            # env.print_grid()

        state = next_state
        step_count += 1

    winner = env.check_winner()

    return {"winner": winner, "steps": step_count, "done": done}


def test_random_vs_random(num_games=50):
    env = GridGameAi()

    results = {"P1": 0, "P2": 0, "draw": 0, "invalid_games": 0}

    for i in range(num_games):
        game = play_random_game(env)

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


env = GridGameAi()

# debug visivo (1 partita)
play_random_game(env, render=True)

# test robustezza
test_random_vs_random(10)
