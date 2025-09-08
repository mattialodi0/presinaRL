import random

class Player:
    def __init__(self):
        self.hand = []
        self.prediction = 0

    def take_hand(self, hand):
        self.hand = hand
        return self.hand

    def make_prediction(self, predictions_made, last=False, round_number=0): # random
        if last:
            if sum(predictions_made) == 5-round_number:
                self.prediction = 1
            else:
                self.prediction = 0
        else:
            self.prediction = sum(1 for card in self.hand if getattr(card, 'suit', None) == 'Denari')
        return self.prediction
    
    def make_prediction_last_round(self, cards, predictions_made): # random
        self.prediction = random.randint(0, 1)
        return self.prediction

    def play_card(self, played_cards): # random
        return self.hand.pop() if self.hand else None