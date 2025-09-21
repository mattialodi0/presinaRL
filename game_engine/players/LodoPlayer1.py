from players.Player import Player
from numpy import average as avg


# to play against good players
class LodoPlayer1(Player):
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
        # self.prediction = self._prob_best_perd(cards_visible=game_state["cards_visible"])
        cards_visible=game_state["cards_visible"]
        predictions_made=game_state["predictions_made"]
        players_position=game_state["players_position"]
        cards_values = [card.value() for card in cards_visible]
        remaining_cards = [v for v in range(0, 40) if v not in cards_values]
        suggested_pred = []
        if max(cards_values) > avg(remaining_cards):
            self.prediction = 0
        else:
            self.prediction = 1

        for p in range(game_state["players_position"].index(self.id)):
            p_pred = predictions_made[players_position.index(p)]
            p_cards_visible = cards_values.copy()
            p_cards_visible.pop(p)
            p_threshold = 19.5 - sum(1 for card in p_cards_visible if card >= 20) + sum(1 for card in p_cards_visible if card < 20)
            #print(p_threshold)
            
            if p_pred > 0:
                if len([c for c in remaining_cards if (c < p_threshold and c > max(cards_values))]) >= \
                    len([c for c in remaining_cards if c < max(cards_values)]):
                    suggested_pred.append(1)
                else:
                    suggested_pred.append(0)
            if p_pred == 0 and not any(v > p_threshold for v in p_cards_visible):
                if len([c for c in remaining_cards if (c >= p_threshold and c < max(cards_values))]) >= \
                    len([c for c in remaining_cards if c > max(cards_values)]):
                    suggested_pred.append(0)
                else:
                    suggested_pred.append(1)
            else:
                suggested_pred.append(0)
        if 1 in suggested_pred:
            self.prediction = 1

        #print(f"card val: {cards_values}")
        #print(f"predictions made: {predictions_made}")
        #print(f"suggested pred: {suggested_pred}")
        return self.prediction

    def play_card(self, game_state):
        return self.hand.pop() if self.hand else None
    

    def _prob_best_perd(self, cards_visible):
        cards_values = [card.value() for card in cards_visible]
        remaining_cards = [v for v in range(0, 40) if v not in cards_values]
        if max(cards_values) > avg(remaining_cards):
            return 0
        else:
            return 1