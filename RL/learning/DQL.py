import numpy as np
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

from game.PresinaEnv import PresinaEnv

env = PresinaEnv(hand_size=4, num_players=4)

# DQN Network
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        neurons = 128
        self.fc1 = nn.Linear(state_size, neurons)
        self.fc2 = nn.Linear(neurons, neurons)
        self.fc3 = nn.Linear(neurons, action_size)
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        # unzip
        states, actions, rewards, next_states, dones = zip(*batch)
        # states and next_states are numpy arrays themselves -> stack into a 2D array
        states = np.stack(states)
        next_states = np.stack(next_states)
        # actions, rewards, dones are scalars -> convert to arrays
        actions = np.array(actions)
        rewards = np.array(rewards)
        dones = np.array(dones)
        return states, actions, rewards, next_states, dones
    def __len__(self):
        return len(self.buffer)

def preprocess_state(state):
    # Flatten dict observation into 1D array
    hand = state['hand']
    played = state['played']
    predictions = np.pad(state['predictions'], (0, env.num_players - 1 - len(state['predictions'])), mode='constant', constant_values=-1)
    print(predictions)
    phase = [state['phase']]
    takes = [state['takes']]
    return np.concatenate([hand, played, predictions, phase, takes]).astype(np.float32)

def select_action(state, policy_net, epsilon, hand_size):
    # Phase 0: prediction, Phase 1: play
    phase = state['phase']
    if phase == 0:
        if random.random() < epsilon:
            predict = random.randint(0, hand_size)
        else:
            state_tensor = torch.tensor(preprocess_state(state)).unsqueeze(0)
            q_values = policy_net(state_tensor)
            predict = int(torch.argmax(q_values).item())
        return {'predict': predict, 'play': 0}
    else:
        valid_cards = [c-1 for c in state['hand'] if c > 0]
        if random.random() < epsilon or not valid_cards:
            play = random.choice(valid_cards) if valid_cards else 0
        else:
            state_tensor = torch.tensor(preprocess_state(state)).unsqueeze(0)
            q_values = policy_net(state_tensor)
            play = int(torch.argmax(q_values).item())
            if play not in valid_cards:
                play = random.choice(valid_cards)
        return {'predict': 0, 'play': play}

def dqn_learning(env, num_episodes, batch_size=64, gamma=0.99, alpha=1e-3, epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=5000, buffer_capacity=10000, target_update=1000):
    obs = env.reset()
    state_size = len(preprocess_state(obs))
    hand_size = env.hand_size
    # Action size: max of predict or play
    action_size = max(hand_size+1, 40)
    policy_net = DQN(state_size, action_size)
    target_net = DQN(state_size, action_size)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    optimizer = optim.Adam(policy_net.parameters(), lr=alpha)
    replay_buffer = ReplayBuffer(buffer_capacity)
    steps_done = 0
    epsilon = epsilon_start

    for i_episode in range(num_episodes):
        state = env.reset()
        done = False
        while not done:
            epsilon = epsilon_end + (epsilon_start - epsilon_end) * np.exp(-1. * steps_done / epsilon_decay)
            action = select_action(state, policy_net, epsilon, hand_size)
            next_state, reward, done, info = env.step(action)
            replay_buffer.push(preprocess_state(state),
                              action['predict'] if state['phase']==0 else action['play'],
                              reward,
                              preprocess_state(next_state),
                              done)
            state = next_state
            steps_done += 1

            # Train
            if len(replay_buffer) >= batch_size:
                states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
                states = torch.tensor(states)
                actions = torch.tensor(actions, dtype=torch.int64)
                rewards = torch.tensor(rewards, dtype=torch.float32)
                next_states = torch.tensor(next_states)
                dones = torch.tensor(dones, dtype=torch.float32)

                q_values = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    next_q_values = target_net(next_states).max(1)[0]
                    expected_q_values = rewards + gamma * next_q_values * (1 - dones)
                loss = nn.functional.mse_loss(q_values, expected_q_values)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Update target network
            if steps_done % target_update == 0:
                target_net.load_state_dict(policy_net.state_dict())

        if (i_episode + 1) % 100 == 0:
            print(f"\rEpisode {i_episode + 1}/{num_episodes}.", end="")
            sys.stdout.flush()

    return policy_net

# TRAIN
policy_net = dqn_learning(env, num_episodes=1000)

# TEST
n_episodes = 1000
wins = losses = errs = 0
hand_size = env.hand_size

for i in range(n_episodes):
    state = env.reset()
    done = False
    while not done:
        phase = state['phase']
        state_tensor = torch.tensor(preprocess_state(state)).unsqueeze(0)
        with torch.no_grad():
            q_values = policy_net(state_tensor)
            if phase == 0:
                action = {'predict': int(torch.argmax(q_values).item()), 'play': 0}
            else:
                valid_cards = [c-1 for c in state['hand'] if c > 0]
                play = int(torch.argmax(q_values).item())
                if play not in valid_cards:
                    play = random.choice(valid_cards) if valid_cards else 0
                action = {'predict': 0, 'play': play}
        next_state, reward, done, info = env.step(action)
        state = next_state

    if reward > 0:
        wins += 1
    elif reward < 0:
        losses += 1
    else:
        errs += 1

print(f"\nPlayed {n_episodes} episodes -> Wins: {wins}, Losses: {losses}, Errors: {errs}")
print(f"Win rate: {wins/n_episodes:.3f}, Loss rate: {losses/n_episodes:.3f}")