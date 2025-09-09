from players.Player import Player
import random


class RandomPlayer(Player):
    def __init__(self, id=0):
        super().__init__(id)

    def make_prediction(self, predictions_made, last=False, hand_size=5): # random
        if last:
            while True:
                self.prediction = random.randint(0, hand_size)
                if not sum(predictions_made)+self.prediction == hand_size:
                    break
        else:
            self.prediction = random.randint(0, hand_size)
        return self.prediction
    
    def make_prediction_last_round(self, cards, predictions_made): # random
        self.prediction = random.randint(0, 1)
        return self.prediction

    def play_card(self, played_cards): # random
        r = random.randint(0, len(self.hand)-1)
        return self.hand.pop(r) if self.hand else None
    