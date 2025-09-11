from players.Player import Player


# to play against random (cannot exploit predictions)
class LodoPlayer(Player):
    def __init__(self, id=0):
        super().__init__(id)

    def take_hand(self, hand):
        super().take_hand(sorted(hand, key=lambda card: (card.suit, card.rank), reverse=True))

    def make_prediction(self, predictions_made, last=False, hand_size=5):
        self.prediction = 0
        for card in self.hand:
            if card.value() > 30:
                self.prediction += 1
        if last:
            if sum(predictions_made)+self.prediction == hand_size:
                if self.prediction > 0:
                    self.prediction -= 1
                else:
                    self.prediction += 1
        return self.prediction

    def make_prediction_last_round(self, cards, predictions_made):
        cards_values = [card.value() for card in cards]
        if max(cards_values) > 20:
            self.prediction = 0
        else:
            self.prediction = 1
        return self.prediction

    def play_card(self, played_cards):
        return self.hand.pop() if self.hand else None


# to play against good players