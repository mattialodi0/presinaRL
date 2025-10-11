import gymnasium as gym
from gymnasium.envs.registration import register
from gymnasium import spaces
import numpy as np
import random

register(
    id='Presina-v2',
    entry_point='PresinaEnv:PresinaEnv',
)

class PresinaEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps":1}
    """
    Improved version with better players
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

    def __init__(self, hand_size=5, num_players=4, agent_pos=0, reward_type='exact', render_mode=None):
        super().__init__()
        assert 2 <= num_players <= 10
        assert 1 <= hand_size <= 5
        assert 0 <= agent_pos < num_players
        self.deck = list(range(1,41))
        self.hand_size = hand_size
        self.num_players = num_players
        self.reward_type = reward_type
        self.agent_pos = agent_pos

        # Observations
        self.observation_space = spaces.Dict({
            'hand': spaces.Box(low=0, high=40, shape=(self.hand_size,), dtype=np.int8),
            'played': spaces.Box(low=0, high=40, shape=(self.num_players,), dtype=np.int8),
            'predictions': spaces.Box(low=0, high=(self.hand_size+1), shape=(self.num_players-1,), dtype=np.int8),
            'phase': spaces.Discrete(2),
            'takes': spaces.Discrete(self.hand_size + 1),
        })

        self.action_space = spaces.Dict({
            'predict': spaces.Discrete(self.hand_size + 1),
            'play': spaces.Discrete(40),
        })

        # internal state
        self.hands = None
        self.played = None
        self.current_phase = 0
        self.agent_prediction = -1
        self.opponent_predictions = None
        self.takes = 0

    def reset(self, seed=None, return_info=False, options=None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        deck = self.deck.copy()
        random.shuffle(deck)
        self.hands = [sorted(deck[i*self.hand_size:(i+1)*self.hand_size]) for i in range(self.num_players)]
        self.played = []
        self.current_phase = 0
        self.agent_prediction = -1
        self.opponent_predictions = []
        self.takes = 0
        self.turn = 0
        
        # predict for opponents before agent
        for pid in range(self.agent_pos):
            pred = make_prediction(self.hands[pid], pid, self.num_players, self.opponent_predictions)
            self.opponent_predictions.append(pred)

        return self._get_obs()

    def step(self, action):
        assert isinstance(action, dict), "Action must be a dict with keys 'predict' and 'play'"
        reward = 0.0
        done = False
        info = {}

        if self.current_phase == 0:
            # predictions for opponents after agent
            for pid in range(self.agent_pos+1, self.num_players):
                pred = make_prediction(self.hands[pid], pid, self.num_players, [self.agent_prediction]+self.opponent_predictions)
                self.opponent_predictions.append(pred)

            p = int(action['predict'])
            if p < 0 or p > self.hand_size:
                p = max(0, min(self.hand_size, p))
                reward -= 0.1
            self.agent_prediction = p
            self.current_phase = 1

            # play for opponents before agent
            for pid in range(self.agent_pos):
                card = random.choice(self.hands[pid])
                self.hands[pid].remove(card)
                self.played.append(card)
            
            return self._get_obs(), reward, done, info
        else:
            play_choice = int(action['play'])
            agent_card = None
            if len(self.hands[self.agent_pos]) == 0:
                # No cards left to play, end episode gracefully
                done = True
                info['error'] = 'Tried to play with empty hand.'
                return self._get_obs(), reward, done, info
            if play_choice+1 in self.hands[self.agent_pos]:
                agent_card = play_choice+1
                self.hands[self.agent_pos].remove(agent_card)
            else:
                agent_card = random.choice(self.hands[self.agent_pos])
                self.hands[self.agent_pos].remove(agent_card)
                reward -= 0.05
                info['invalid_play_replaced_with'] = agent_card
            self.played.append(agent_card)
            
            for pid in range(self.agent_pos+1, self.num_players):
                card = random.choice(self.hands[pid])
                self.hands[pid].remove(card)
                self.played.append(card)

            winner = np.argmax(self.played)
            if winner == self.agent_pos:
                self.takes += 1
            self.turn += 1

            if self.turn < self.hand_size:
                self.played = []
                # play for opponents before agent
                for pid in range(self.agent_pos):
                    card = random.choice(self.hands[pid])
                    self.hands[pid].remove(card)
                    self.played.append(card)
                return self._get_obs(), reward, False, info

            done = True
            actual = self.takes
            predicted = self.agent_prediction

            reward += 1.0 if actual == predicted else abs(actual - predicted)*-1

            info.update({'actual': actual, 'predicted': predicted})
            return self._get_obs(), reward, done, info

    def render(self, mode='human'):
        print(f"Phase: {self.current_phase}, prediction: {self.agent_prediction}, takes: {self.takes}")
        print(f"Agent hand: {self.hands[0]}")

    def _get_obs(self):
        hand_arr = np.zeros(self.hand_size, dtype=np.int8)
        for i, c in enumerate(sorted(self.hands[self.agent_pos])):
            hand_arr[i] = c
        played_arr = np.array(self.played, dtype=np.int8)
        preds_arr = np.array(self.opponent_predictions, dtype=np.int8)
        return {
            'hand': hand_arr,
            'played': played_arr,
            'predictions': preds_arr,
            'phase': self.current_phase,
            'takes': self.takes,
        }
    

def make_prediction(hand, player_position, num_players, predictions_made):
    prediction = 0
    for card in hand:
        if card > 30:
            prediction += 1
    if player_position == num_players - 1:
        if sum(predictions_made) + prediction == len(hand):
            if prediction > 0:
                prediction -= 1
            else:
                prediction += 1
    return prediction