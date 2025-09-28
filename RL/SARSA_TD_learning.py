import gym
import matplotlib
import numpy as np
import sys

from collections import defaultdict
if "../" not in sys.path:
  sys.path.append("../") 
from game.presina_env import PresinaEnv

env = PresinaEnv()

def make_epsilon_greedy_policy(Q, epsilon, nA):
    """Creates an epsilon-greedy policy based on a given Q-function and epsilon."""
    def policy_fn(observation):
        A = np.ones(nA, dtype=float) * epsilon / nA
        best_action = np.argmax(Q[observation])
        A[best_action] += (1.0 - epsilon)
        return A
    return policy_fn


def sarsa(env, num_episodes, discount_factor=1.0, alpha=0.5, epsilon=0.1):
    """SARSA algorithm: On-policy TD control. Finds the optimal epsilon-greedy policy."""
    
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
        action_probs = policy(state)
        action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
        
        # One step in the environment
        while True:
            # Take a step
            next_state, reward, done, _, _ = env.step(action)
            next_state = tuple(sorted(next_state))
            # Pick the next action
            next_action_probs = policy(next_state)
            next_action = np.random.choice(np.arange(len(next_action_probs)), p=next_action_probs)
                        
            # TD Update
            td_target = reward + discount_factor * Q[next_state][next_action]
            td_delta = td_target - Q[state][action]
            Q[state][action] += alpha * td_delta
    
            if done:
                break
                
            action = next_action
            state = next_state        
    
    return Q


# TRAIN
Q = sarsa(env, 100000)

# TEST
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