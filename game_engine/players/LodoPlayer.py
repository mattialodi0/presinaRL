from players.Player import Player
from numpy import average as avg


# to play against random (cannot exploit predictions)
class LodoPlayer(Player):
    "Lodo player class, all decisions are based on simple heuristics."

    def __init__(self, id=0):
        super().__init__(id)

    def take_hand(self, hand):
        super().take_hand(sorted(hand, key=lambda card: (card.suit, card.rank), reverse=True))

    def make_prediction(self, game_state):
        self.prediction = 0
        for card in self.hand:
            if card.value() > 30:
                self.prediction += 1
        if game_state["players_position"].index(self.id) == game_state["num_players"] - 1:
            if sum(game_state["predictions_made"])+self.prediction == game_state["hand_size"]:
                if self.prediction > 0:
                    self.prediction -= 1
                else:
                    self.prediction += 1
        return self.prediction

    def make_prediction_last_round(self, game_state):
        cards_values = [card.value() for card in game_state["cards_visible"]]
        remaining_cards = [v for v in range(0, 41) if v not in cards_values]
        if max(cards_values) > avg(remaining_cards):
            self.prediction = 0
        else:
            self.prediction = 1
        return self.prediction

    def play_card(self, game_state):
        return self.hand.pop() if self.hand else None


# to play against good players