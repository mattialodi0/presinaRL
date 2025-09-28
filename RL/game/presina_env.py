import gymnasium as gym
from gymnasium.envs.registration import register
import numpy as np
import random

register(
    id='Presina-v0',
    entry_point='presina_env:PresinaEnv',
)

class PresinaEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps":1}

    def __init__(self, n_players=4, render_mode=None):
        super().__init__()
        self.render_mode = render_mode
        self.n_players = n_players
        self.action_space = gym.spaces.Discrete(2)  # 0: Don't take the card, 1: Take the card
        self.observation_space = gym.spaces.Box(low=1, high=40, shape=(self.n_players-1,), dtype=np.int32)
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.player_cards = self.deal_cards()
        self.player_predictions = [0 for _ in range(self.n_players)]
        self.current_player = 0
        self.done = False

        c = self.player_cards.copy()
        c.pop(self.current_player)
        c.sort()
        obs = np.array(c)
        info = {}
        return obs, info
    

    def step(self, action):
        self.done = True

        # make player action
        if action == 1:
            self.player_predictions[self.current_player] = 1
        # simulate other random players
        for i in range(self.n_players):
            if i != self.current_player:
                self.player_predictions[i] = random.choice([0, 1])
        
        reward = self.calculate_reward()

        c = self.player_cards.copy()
        c.pop(self.current_player)
        c.sort()
        obs = np.array(c)
        info = {}

        if(self.render_mode=='human'):
            self.render()
            print(f"Action: {action}, Reward: {reward}")

        return obs, reward, self.done, False, info


    def render(self, mode='human'):
        print(f"Current cards: {self.player_cards}")

    def deal_cards(self):
        # Simulate dealing cards (for simplicity, using random integers)
        return [random.randint(1, 40) for _ in range(self.n_players)]

    def calculate_reward(self):
        highest_card = max(self.player_cards)
        if self.player_predictions[self.current_player] == 1:
            if self.player_cards[self.current_player] == highest_card:
                return 1
            else:
                return -1
        elif self.player_predictions[self.current_player] == 0:
            if self.player_cards[self.current_player] != highest_card:
                return 1
            else:
                return -1
        else:
            return 0