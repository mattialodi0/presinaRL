import gym
from gym import spaces
import numpy as np
from ..game_engine.Game import Game 



class PresinaEnv(gym.Env):
    """
    Custom Environment for your card game.
    Each round = (1) prediction phase + (2) play card phase.
    """

    metadata = {"render.modes": ["human"]}

    def __init__(self, num_players=4, num_rounds=5, obs_dim=50):
        super(PresinaEnv, self).__init__()

        self.num_players = num_players
        self.num_rounds = num_rounds

        # Your Game object
        self.game = None

        # Phase alternates between "prediction" and "card"
        self.phase = "prediction"

        # Example spaces (you may adjust depending on rules)
        self.num_predictions = 5   # e.g., max predictions allowed
        self.max_hand_size = 5     # cards in hand

        self.action_space = spaces.Discrete(
            max(self.num_predictions, self.max_hand_size)
        )

        # Observation = encoded game state
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(obs_dim,), dtype=np.float32
        )

        # Track current round/hand
        self.current_round = 0
        self.current_hand_size = 0
        self.done = False

    def reset(self):
        """Start a fresh game."""
        self.game = Game(n=self.num_players, r=self.num_rounds)
        self.current_round = 0
        self.phase = "prediction"
        self.done = False

        return self._get_obs()

    def step(self, action):
        """Apply RL agent's action depending on phase."""
        reward = 0.0

        if self.phase == "prediction":
            # Apply prediction for player 0
            self._apply_prediction(action)
            self.phase = "card"
        else:  # card phase
            self._apply_card(action)
            reward, self.done = self._evaluate_round()
            self.phase = "prediction"

        return self._get_obs(), reward, self.done, {}

    def _apply_prediction(self, action):
        """Plug into your game's prediction logic for player 0."""
        # TODO: integrate with self.game.makePredictions
        pass

    def _apply_card(self, action):
        """Play a card from RL agent's hand."""
        # TODO: integrate with self.game.playCards
        pass

    def _evaluate_round(self):
        """Decide reward + check game end after a full round."""
        # TODO: use self.game.determineCatches, checkForErrors
        reward = 0.0
        done = (self.current_round >= self.num_rounds)
        return reward, done

    def _get_obs(self):
        """Turn game state into an observation vector."""
        # Example: encode agent’s hand, scores, round number
        obs = np.zeros(self.observation_space.shape, dtype=np.float32)

        # TODO: fill with features from self.game
        # e.g., self.game.predictions, self.game.errors, hand cards, etc.

        return obs

    def render(self, mode="human"):
        print(f"Round {self.current_round}, Phase: {self.phase}")
        # Optionally show hands, predictions, scores

    def close(self):
        pass