import gym
import matplotlib
import numpy as np
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

from game.presina_env import PresinaEnv

env = PresinaEnv()

# DQN Network
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_size)
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
        state, action, reward, next_state, done = map(np.array, zip(*batch))
        return state, action, reward, next_state, done
    def __len__(self):
        return len(self.buffer)

def preprocess_state(state):
    # Flatten and normalize state for NN input
    return np.array(state, dtype=np.float32)

def select_action(state, policy_net, epsilon, action_size):
    if random.random() < epsilon:
        return random.randrange(action_size)
    with torch.no_grad():
        state_tensor = torch.tensor(preprocess_state(state)).unsqueeze(0)
        q_values = policy_net(state_tensor)
        return int(torch.argmax(q_values).item())

def dqn_learning(env, num_episodes, batch_size=64, gamma=0.99, alpha=1e-3, epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=5000, buffer_capacity=10000, target_update=1000):
    state_size = len(preprocess_state(env.reset()[0]))
    action_size = env.action_space.n
    policy_net = DQN(state_size, action_size)
    target_net = DQN(state_size, action_size)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    optimizer = optim.Adam(policy_net.parameters(), lr=alpha)
    replay_buffer = ReplayBuffer(buffer_capacity)
    steps_done = 0
    epsilon = epsilon_start

    for i_episode in range(num_episodes):
        state, info = env.reset()
        state = tuple(sorted(state))
        done = False
        while not done:
            epsilon = epsilon_end + (epsilon_start - epsilon_end) * np.exp(-1. * steps_done / epsilon_decay)
            action = select_action(state, policy_net, epsilon, action_size)
            next_state, reward, done, _, _ = env.step(action)
            next_state = tuple(sorted(next_state))
            replay_buffer.push(preprocess_state(state), action, reward, preprocess_state(next_state), done)
            state = next_state
            steps_done += 1

            # Train
            if len(replay_buffer) >= batch_size:
                states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
                states = torch.tensor(states)
                actions = torch.tensor(actions)
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
policy_net = dqn_learning(env, num_episodes=10000)

# TEST
n_episodes = 1000
wins = losses = errs = 0
action_size = env.action_space.n

for i in range(n_episodes):
    state, info = env.reset()
    state = tuple(sorted(state))
    done = False
    while not done:
        state_tensor = torch.tensor(preprocess_state(state)).unsqueeze(0)
        with torch.no_grad():
            q_values = policy_net(state_tensor)
            action = int(torch.argmax(q_values).item())
        next_state, reward, done, _, _ = env.step(action)
        next_state = tuple(sorted(next_state))
        state = next_state

    if reward > 0:
        wins += 1
    elif reward < 0:
        losses += 1
    else:
        errs += 1

print(f"\nPlayed {n_episodes} episodes -> Wins: {wins}, Losses: {losses}, Errors: {errs}")
print(f"Win rate: {wins/n_episodes:.3f}, Loss rate: {losses/n_episodes:.3f}")