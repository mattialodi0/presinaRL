from players.Player import Player
import random


class RandomPlayer(Player):
    """Random player class, all decisions are random."""
    def __init__(self, id=0):
        super().__init__(id)
    
    def make_prediction(self, game_state):
        if game_state["players_position"].index(self.id) == game_state["num_players"] - 1:
            while True:
                self.prediction = random.randint(0, game_state["hand_size"])
                if not sum(game_state["predictions_made"])+self.prediction == game_state["hand_size"]:
                    break
        else:
            self.prediction = random.randint(0, game_state["hand_size"])
        return self.prediction
    
    def make_prediction_last_round(self, game_state):
        self.prediction = random.randint(0, 1)
        return self.prediction

    def play_card(self, game_state):
        r = random.randint(0, len(self.hand)-1)
        return self.hand.pop(r) if self.hand else None