import numpy as np
from collections import defaultdict
import sys
if "../" not in sys.path:
  sys.path.append("../") 
from RL.game.PresinaEnvMultiAgent import PresinaEnvMultiAgent



# Parameters
num_episodes = 10000
hand_size = 4
num_players = 4
epsilon = 0.1
gamma = 1.0

env = PresinaEnvMultiAgent(hand_size=hand_size, num_players=num_players)
# Shared Q-tables for all agents: dicts
Q_pred = defaultdict(lambda: np.full(env.hand_size + 1, -np.inf))
Q_play = defaultdict(lambda: np.full(env.hand_size, -np.inf))

returns_predict = defaultdict(list)  # returns_predict[(state, action)] = [G, ...]
returns_play = defaultdict(list)     # returns_play[(state, action)] = [G, ...]

def epsilon_greedy(Q, state, epsilon):
    q = Q[state]
    if np.random.rand() < epsilon:
        return np.random.randint(len(q))
    else:
        return np.argmax(q)

for episode in range(num_episodes):
    if (episode+1) % 1000 == 0:
        print(f"\rEpisode {episode+1}/{num_episodes}.", end="")
        sys.stdout.flush()

    obs = env.reset()
    done = False
    episode_history = [[] for _ in range(num_players)]  # [(state, action, reward)]
    while not done:
        phase = obs['phase']
        pid = obs['turn']
        if phase == 0:
            # Prediction phase
            state = tuple(obs['hand'])
            action = epsilon_greedy(Q_pred, state, epsilon)
            next_obs, reward, done, info = env.step({'predict': action})
            episode_history[pid].append((('predict', state), action, reward))
        else:
            # Play phase
            hand = obs['hand']
            valid_cards = [i for i, c in enumerate(hand) if c > 0]
            state = tuple(hand)
            if valid_cards:
                action = np.random.choice(valid_cards) if np.random.rand() < epsilon else np.argmax(Q_play[state])
            else:
                action = 0
            next_obs, reward, done, info = env.step({'play': action})
            episode_history[pid].append((('play', state), action, reward))
        obs = next_obs

    # Monte Carlo update
    for pid in range(num_players):
        G = 0
        for (stype, state), action, reward in reversed(episode_history[pid]):
            G = gamma * G + reward[pid]
            key = (state, action)
            if stype == 'predict':
                returns_predict[key].append(G)
                Q_pred[state][action] = np.mean(returns_predict[key])
            else:
                returns_play[key].append(G)
                Q_play[state][action] = np.mean(returns_play[key])


# Test phase
test_episodes = 1000
total_rewards = np.zeros(num_players)
total_wins = np.zeros(num_players)
for episode in range(test_episodes):
    obs = env.reset()
    done = False
    while not done:
        phase = obs['phase']
        pid = obs['turn']
        if phase == 0:
            state = tuple(obs['hand'])
            action = np.argmax(Q_pred[state])
            next_obs, reward, done, info = env.step({'predict': action})
        else:
            hand = obs['hand']
            valid_cards = [i for i, c in enumerate(hand) if c > 0]
            state = tuple(hand)
            action = np.argmax(Q_play[state]) if valid_cards else 0
            next_obs, reward, done, info = env.step({'play': action})
        obs = next_obs
        for pid in range(num_players):
            total_rewards[pid] += reward[pid]
            if reward[pid] > 0:
                total_wins[pid] += 1

print("")
print("Average rewards per agent:", total_rewards / test_episodes)
print("Total wins per agent:", total_wins)


# demostration match
# obs = env.reset()
# env.render()
# done = False
# while not done:
#     phase = obs['phase']
#     pid = obs['turn']
#     if phase == 0:
#         state = tuple(obs['hand'])
#         action = np.argmax(Q_pred[state])
#         next_obs, reward, done, info = env.step({'predict': action})
#         env.render()
#     else:
#         hand = obs['hand']
#         valid_cards = [i for i, c in enumerate(hand) if c > 0]
#         state = tuple(hand)
#         action = np.argmax(Q_play[state]) if valid_cards else 0
#         next_obs, reward, done, info = env.step({'play': action})
#         env.render()
#     obs = next_obs