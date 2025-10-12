import sys
if "../" not in sys.path:
  sys.path.append("../") 
from RL.game.PresinaEnvMultiAgent import PresinaEnvMultiAgent

env = PresinaEnvMultiAgent(hand_size=4, num_players=4)
obs = env.reset()
env.render()

done = False
while not done:
    if obs['phase'] == 0:
        # Prediction phase: choose a random prediction for each player
        action = {'predict': 2}  # Example: always predict 2
    else:
        # Play phase: play the lowest card in hand
        hand = obs['hand']
        valid_cards = [c for c in hand if c > 0]
        if valid_cards:
            play_card = min(valid_cards) - 1  # index for action
        else:
            play_card = 0
        action = {'play': play_card}
    obs, reward, done, info = env.step(action)
    env.render()
    print(f"")
    # print(f"Reward: {reward}, Info: {info}")