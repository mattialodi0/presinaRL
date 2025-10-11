import numpy as np
import sys

from collections import defaultdict
if "../" not in sys.path:
  sys.path.append("../") 
from RL.game.PresinaEnv import PresinaEnv

env = PresinaEnv(hand_size=4, num_players=4, agent_pos=3)

# For PresinaEnv, actions are dicts. We need to enumerate possible actions for each phase.
def get_possible_actions(obs, hand_size):
    if obs['phase'] == 0:
        # Prediction phase: choose prediction in [0, hand_size]
        return [{'predict': p, 'play': 0} for p in range(hand_size+1)]
    else:
        # Play phase: choose a card from hand
        valid_cards = [c for c in obs['hand'] if c > 0]
        return [{'predict': 0, 'play': c-1} for c in valid_cards]

def obs_to_state(obs):
    # Convert observation dict to a 4D tuple for Q-table key (hand, played, predictions, takes)
    return (
        tuple(sorted(obs['hand'])),
        tuple(sorted(obs['played'])) if obs['phase'] == 1 else tuple([-1]*4),
        tuple(sorted(obs['predictions'])) if obs['phase'] == 1 else tuple([-1]*3),
        obs['takes'] if obs['phase'] == 1 else 0,
    )

r = 0
def make_epsilon_greedy_policy(Q_pred, Q_play, epsilon, env):
    def policy_fn(obs):
        global r
        state = obs_to_state(obs)
        actions = get_possible_actions(obs, env.hand_size)
        nA = len(actions)
        if nA > 1:
            A = np.ones(nA, dtype=float) * epsilon / nA
            # Select Q table based on phase
            Q = Q_pred if obs['phase'] == 0 else Q_play
            if state in Q:
                best_action = np.argmax(Q[state])
                if best_action < nA:
                    A[best_action] += (1.0 - epsilon)
                else:
                    A[0] += (1.0 - epsilon)
                r += 1
            else:
                rand_action = np.random.randint(nA)
                A[rand_action] += (1.0 - epsilon)
            return A, actions
        else:
            return np.array([1.0]), actions
    return policy_fn

def mc_control_epsilon_greedy(env, num_episodes, discount_factor=1.0, epsilon=0.1):
    try:
        returns_sum_pred = defaultdict(float)
        returns_count_pred = defaultdict(float)
        returns_sum_play = defaultdict(float)
        returns_count_play = defaultdict(float)
        Q_pred = defaultdict(lambda: np.full(env.hand_size + 1, -np.inf))  # max possible actions per state
        Q_play = defaultdict(lambda: np.full(env.hand_size, -np.inf))
        policy = make_epsilon_greedy_policy(Q_pred, Q_play, epsilon, env)

        for i_episode in range(1, num_episodes + 1):
            if i_episode % 1000 == 0:
                print(f"\rEpisode {i_episode}/{num_episodes}.", end="")
                sys.stdout.flush()

            episode = []
            obs = env.reset()
            if isinstance(obs, tuple):
                obs = obs[0]
            state = obs_to_state(obs)
            for t in range(env.hand_size + 2):
                A_probs, actions = policy(obs)
                nA = len(actions)
                if nA == 0:
                    break
                action_idx = np.random.choice(np.arange(nA), p=A_probs)
                action = actions[action_idx]
                next_obs, reward, done, info = env.step(action)
                if isinstance(next_obs, tuple):
                    next_obs = next_obs[0]
                next_state = obs_to_state(next_obs)
                episode.append((state, action_idx, reward, actions, obs['phase']))
                if done:
                    break
                obs = next_obs
                state = next_state

            # Find all (state, action_idx, phase) pairs we've visited in this episode
            sa_in_episode = set([(x[0], x[1], x[4]) for x in episode])
            for state, action_idx, phase in sa_in_episode:
                sa_pair = (state, action_idx)
                first_occurence_idx = next(i for i,x in enumerate(episode)
                                        if x[0] == state and x[1] == action_idx and x[4] == phase)
                G = sum([x[2]*(discount_factor**i) for i,x in enumerate(episode[first_occurence_idx:])])
                if phase == 0:
                    returns_sum_pred[sa_pair] += G
                    returns_count_pred[sa_pair] += 1.0
                    Q_pred[state][action_idx] = returns_sum_pred[sa_pair] / returns_count_pred[sa_pair]
                else:
                    returns_sum_play[sa_pair] += G
                    returns_count_play[sa_pair] += 1.0
                    Q_play[state][action_idx] = returns_sum_play[sa_pair] / returns_count_play[sa_pair]
    except Exception as e:
        print("")
        print(e)
        print("")
    finally:
        return Q_pred, Q_play, policy


# TRAIN
Q_pred, Q_play, policy = mc_control_epsilon_greedy(env, num_episodes=100000, epsilon=0.1)

# print("Q table indexes example:")
# print(list(Q.keys())[0])

# TEST
n_episodes = 1000
wins = losses = errs = 0
tot_reward = 0

for i in range(n_episodes):
    obs = env.reset()
    if isinstance(obs, tuple):
        obs = obs[0]
    state = obs_to_state(obs)
    done = False
    while not done:
        key = state
        actions = get_possible_actions(obs, env.hand_size)
        nA = len(actions)
        # Select Q table based on phase
        Q = Q_pred if obs['phase'] == 0 else Q_play
        if key in Q and nA > 0:
            action_idx = int(np.argmax(Q[key][:nA]))
            action = actions[action_idx]
        elif nA > 0:
            A_probs, actions = make_epsilon_greedy_policy(Q_pred, Q_play, 0.1, env)(obs)
            action_idx = np.random.choice(np.arange(nA), p=A_probs)
            action = actions[action_idx]
        else:
            break
        next_obs, reward, done, info = env.step(action)
        if isinstance(next_obs, tuple):
            next_obs = next_obs[0]
        state = obs_to_state(next_obs)
        obs = next_obs

    if reward > 0:
        wins += 1
    elif reward < 0:
        losses += 1
    else:
        errs += 1

    tot_reward += reward

print("")
print(f"Played {n_episodes} episodes -> Wins: {wins}, Losses: {losses}, Errors: {errs}")
print(f"Win rate: {wins/n_episodes:.3f}, Loss rate: {losses/n_episodes:.3f}")

print(f"Average reward per episode: {tot_reward/n_episodes:.3f}")
print(r)