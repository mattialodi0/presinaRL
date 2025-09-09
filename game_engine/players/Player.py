import random


class Player:
    def __init__(self, id=0):
        self.hand = []
        self.prediction = 0
        self.id = id


    def take_hand(self, hand):
        self.hand = hand
        return self.hand

    def make_prediction(self, predictions_made, last=False, hand_size=5): # random
        if last:
            if sum(predictions_made) == hand_size:
                self.prediction = 1
            else:
                self.prediction = 0
        else:
                self.prediction = 0
        return self.prediction
    
    def make_prediction_last_round(self, cards, predictions_made): # random
        self.prediction = 0
        return self.prediction

    def play_card(self, played_cards): # random
        return self.hand.pop() if self.hand else None
    