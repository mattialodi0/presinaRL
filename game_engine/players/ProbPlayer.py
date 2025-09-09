from Player import Player


class ProbPlayer(Player):
    def __init__(self, id=0):
        super().__init__(id)

    def make_prediction(self, predictions_made, last=False, hand_size=5):
        return self.prediction

    def make_prediction_last_round(self, cards, predictions_made):
        return self.prediction

    def play_card(self, played_cards):
        return self.hand.pop() if self.hand else None
