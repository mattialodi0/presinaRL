from pettingzoo.utils.env import AECEnv
from gymnasium import spaces
import numpy as np
import random

class PresinaEnvMultiAgent(AECEnv):
    metadata = {"render_modes": ["human"], "render_fps":1}
    """
    Multi agent version
    Rules:
      - Deck of integers 1..40.
      - num_players players total; agent is player 0.
      - At reset each player gets `hand_size` cards.
      - Phase 0: prediction phase. Agent chooses an integer prediction p in [0,hand_size]
      - Phase 1: play phase. For hand_size turns, each player plays one card per turn.
        The highest card played wins the turn and that player "takes" for that turn.
      - Opponents play uniformly at random from their remaining hand.
      - Episode ends after all turns. Reward is computed from the agent's prediction vs actual
        number of turns taken.
    Observation (Dict):
      - 'hand': array of length hand_size with integers representing the agent's current hand (0 padded)
      - 'played': array of length num_players with the most recently played cards (0 if not yet)
      - 'predictions': array of length num_players-1 with other players' predictions
      - 'phase': scalar (0 = prediction phase, 1 = play phase)
      - 'takes': scalar = cards taken so far by the agent
    Action (Dict):
      - 'predict': Discrete(hand_size+1) -- valid only in phase 0
      - 'play': Discrete(40) -- index (card value-1) to play; invalid plays (not in hand)
                   will be handled (see code) by selecting a random valid card and applying
                   a small penalty.
    """

    def __init__(self, hand_size=5, num_players=4, reward_type='exact', render_mode=None):
        super().__init__()
        assert 2 <= num_players <= 10
        assert 1 <= hand_size <= 5
        self.deck = list(range(1,41))
        self.hand_size = hand_size
        self.num_players = num_players
        self.reward_type = reward_type

        # Turn-based: only one agent acts per step
        self.observation_space = spaces.Dict({
            'hand': spaces.Box(low=0, high=40, shape=(self.hand_size,), dtype=np.int8),
            'played': spaces.Box(low=0, high=40, shape=(self.num_players,), dtype=np.int8),
            'predictions': spaces.Box(low=0, high=(self.hand_size+1), shape=(self.num_players,), dtype=np.int8),
            'takes': spaces.Discrete(self.hand_size + 1),
            'phase': spaces.Discrete(2),
            'turn': spaces.Discrete(self.num_players),
        })
        self.action_space = spaces.Dict({
            'predict': spaces.Discrete(self.hand_size + 1),
            'play': spaces.Discrete(40),
        })

        # internal state
        self.hands = None
        self.played = None
        self.current_phase = 0
        self.predictions = None
        self.takes = None
        self.turn = 0
        self.turn_step = 0
        self.h=0
        self.k=0

    def reset(self, seed=None, return_info=False, options=None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        deck = self.deck.copy()
        random.shuffle(deck)
        self.hands = [sorted(deck[i*self.hand_size:(i+1)*self.hand_size]) for i in range(self.num_players)]
        self.played = []
        self.current_phase = 0
        self.predictions = [-1 for _ in range(self.num_players)]
        self.takes = [0 for _ in range(self.num_players)]
        self.turn = 0
        self.turn_step = 0
        return self._get_obs_turn()

    def step(self, action):
        # Only the current agent acts
        reward = [0.0 for _ in range(self.num_players)]
        done = False
        info = {}
        pid = self.turn_step

        # Prediction phase
        if self.current_phase == 0:            
            p = int(action['predict'])
            if p < 0 or p > self.hand_size:
                p = max(0, min(self.hand_size, p))
                reward[pid] -= 0.1
            self.predictions[pid] = p
            self.turn_step += 1
            # switch to play phase
            if self.turn_step >= self.num_players:
                self.turn_step = 0
                self.current_phase = 1
            return self._get_obs_turn(), reward, done, info

        # Play phase
        else:
            # print(pid)
            # print(self.hands)
            # print(action)
            play_choice = int(action['play'])
            if 0 <= play_choice and play_choice < self.hand_size:
                self.h += 1
                card = self.hands[pid][play_choice]
                self.hands[pid].remove(card)
            else:
                self.k += 1
                card = random.choice(self.hands[pid])
                self.hands[pid].remove(card)
                reward[pid] -= 0.05
                info['invalid_play_replaced_with'] = card
            if self.turn_step == 0:
                self.played = [0 for _ in range(self.num_players)]
            self.played[pid] = card
            self.turn_step += 1

            # start new turn
            if self.turn_step >= self.num_players:
                self.turn_step = 0
                self.turn += 1
                winner = np.argmax(self.played)
                self.takes[winner] += 1

                # Check if game over
                if len(self.hands[0]) == 0 or self.turn >= self.hand_size:
                    done = True
                    actual = self.takes[pid]
                    predicted = self.predictions[pid]
                    if actual == predicted:
                        reward[pid] += 1.0
                    else:
                        reward[pid] += abs(actual - predicted)*-1
                    info.update({'actual': actual, 'predicted': predicted})

            return self._get_obs_turn(), reward, done, info

    def render(self, mode='human'):
        print(f"Phase: {self.current_phase}, turn: {self.turn}, step: {self.turn_step}")
        for pid in range(self.num_players):
            print(f"Player {pid}: hand={self.hands[pid]}, prediction={self.predictions[pid]}, takes={self.takes[pid]}")
        print(self.h)
        print(self.k)

    def _get_obs_turn(self):
        # Only return observation for current agent
        pid = self.turn_step
        hand_arr = np.zeros(self.hand_size, dtype=np.int8)
        for i, c in enumerate(sorted(self.hands[pid])):
            hand_arr[i] = c
        played_arr = np.array(self.played, dtype=np.int8) if self.played else np.zeros(self.num_players, dtype=np.int8)
        preds_arr = np.array(self.predictions, dtype=np.int8)
        obs = {
            'hand': hand_arr,
            'played': played_arr,
            'predictions': preds_arr,
            'phase': self.current_phase,
            'takes': self.takes[pid],
            'turn': pid,
        }
        return obs
