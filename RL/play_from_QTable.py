import numpy as np
from collections import defaultdict
import sys
import pickle
if "../" not in sys.path:
  sys.path.append("../") 
from RL.game.PresinaEnvMultiAgent import PresinaEnvMultiAgent


Q_pred, Q_play = None, None
with open("Q_pred.pkl", "rb") as f:
    Q_pred = pickle.load(f)
with open("Q_play.pkl", "rb") as f:
    Q_play = pickle.load(f)


env = PresinaEnvMultiAgent(hand_size=4, num_players=4)
obs = env.reset()
next_obs = None
# env.render()
done = False
i = 0
while not done:
    print(obs)
    phase = obs['phase']
    pid = env.turn_step
    if pid == 0:
        if phase == 0:
            action = int(input("Your turn to predict (0-4): "))
            next_obs, reward, done, info = env.step({'predict': action})
        else:
            action = int(input(f"Your turn to play (0-{sum([1 for i in obs['hand'] if i > 0]) - 1}): "))
            next_obs, reward, done, info = env.step({'play': action})
    else:
        if phase == 0:
            try:
                state = tuple(obs['hand'])
                action = np.argmax(Q_pred[state])
            except Exception as e:
                action = np.random.choice(range(len(obs['hand'])))
            next_obs, reward, done, info = env.step({'predict': action})
        else:
            hand = obs['hand']
            valid_cards = [i for i, c in enumerate(hand) if c > 0]
            try:
                state = tuple(hand)
                action = np.argmax(Q_play[state]) if valid_cards else 0
            except Exception as e:
                action = np.random.choice(valid_cards) if valid_cards else 0
            next_obs, reward, done, info = env.step({'play': action})
    obs = next_obs
    i += 1
    # env.render()
    print(reward)
    print("---")