import gym
import matplotlib
import numpy as np
import sys

from collections import defaultdict
if "../" not in sys.path:
  sys.path.append("../") 
from RL.game.PresinaEnvLastRound import PresinaEnvLastRound

env = PresinaEnvLastRound()

def make_epsilon_greedy_policy(Q, epsilon, nA):
    """Creates an epsilon-greedy policy based on a given Q-function and epsilon."""
    def policy_fn(observation):
        A = np.ones(nA, dtype=float) * epsilon / nA
        best_action = np.argmax(Q[observation])
        A[best_action] += (1.0 - epsilon)
        return A
    return policy_fn

def q_learning(env, num_episodes, discount_factor=1.0, alpha=0.5, epsilon=0.1):
    """
    Q-Learning algorithm: Off-policy TD control. Finds the optimal greedy policy
    while following an epsilon-greedy policy
    
    Args:
        env: OpenAI environment.
        num_episodes: Number of episodes to run for.
        discount_factor: Gamma discount factor.
        alpha: TD learning rate.
        epsilon: Chance to sample a random action. Float between 0 and 1.
    
    Returns:
        A tuple (Q, episode_lengths).
        Q is the optimal action-value function, a dictionary mapping state -> action values.
        stats is an EpisodeStats object with two numpy arrays for episode_lengths and episode_rewards.
    """
    
    # The final action-value function.
    # A nested dictionary that maps state -> (action -> action-value).
    Q = defaultdict(lambda: np.zeros(env.action_space.n))

    # The policy we're following
    policy = make_epsilon_greedy_policy(Q, epsilon, env.action_space.n)
    
    for i_episode in range(num_episodes):
        # Print out which episode we're on, useful for debugging.
        if (i_episode + 1) % 100 == 0:
            print("\rEpisode {}/{}.".format(i_episode + 1, num_episodes), end="")
            sys.stdout.flush()
        
        # Reset the environment and pick the first action
        state, info = env.reset()
        state = tuple(sorted(state))
        
        # One step in the environment
        # total_reward = 0.0
        while True:
            
            # Take a step
            action_probs = policy(state)
            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
            next_state, reward, done, _, _ = env.step(action)
            next_state = tuple(sorted(next_state))
            
            # TD Update
            best_next_action = np.argmax(Q[next_state])    
            td_target = reward + discount_factor * Q[next_state][best_next_action]
            td_delta = td_target - Q[state][action]
            Q[state][action] += alpha * td_delta
                
            if done:
                break
                
            state = next_state
    
    return Q


# TRAIN
Q = q_learning(env, 100000)

# TEST
# n_episodes = 10
# ep_len = []
# policy = make_epsilon_greedy_policy(Q, 0.1, env.action_space.n)

# for i in range(n_episodes):
#     state = env.reset()
#     done = False
#     l = 0
#     while not done:
#         # action_probs = policy(state)
#         # action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
#         # state, reward, done, _ = env.step(action)
        
#         action = np.argmax(Q[state])
#         state, reward, done, _ = env.step(action)

#         l += 1

#     ep_len.append(l)

# print(f"Played {n_episodes} episodes -> Average episode length: {np.mean(ep_len):.2f}")
# print(f"Median episode length: {np.median(ep_len):.2f}, Min episode length: {np.min(ep_len):.2f}, Max episode length: {np.max(ep_len):.2f}")


n_episodes = 1000
wins = losses = errs = 0
policy = make_epsilon_greedy_policy(Q, 0.1, env.action_space.n)

for i in range(n_episodes):
    state, info = env.reset()
    state = tuple(sorted(state))
    done = False
    while not done:
        key = state
        if key in Q:
            action = int(np.argmax(Q[key]))  # greedy action from learned Q
        else:
            probs = policy(state)
            action = np.random.choice(np.arange(len(probs)), p=probs)
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