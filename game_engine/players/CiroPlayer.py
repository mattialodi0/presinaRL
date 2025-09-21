from players.Player import Player
from numpy import average as avg


# to play against random (cannot exploit predictions)
class CiroPlayer(Player):
    "Ciro player class, all decisions are based on heuristics."

    def __init__(self, id=0):
        super().__init__(id)


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
        remaining_cards = [v for v in range(0, 40) if v not in cards_values]
        if max(cards_values) > avg(remaining_cards):
            self.prediction = 0
        else:
            self.prediction = 1
        return self.prediction

    def play_card(self, game_state):
        s = lambda c: c.value()
        preds, catches = game_state["predictions"], game_state["catches"]
        played, order = game_state.get("played_cards", {}), game_state["players_position"]
        myposition, H = self.id, self.hand
        H_ordinata = sorted(H, key=s)                # bassa→alta
        opening = not played
        necessità_presa = int(catches[myposition]) < int(self.prediction or 0)
        curmax = max((s(c) for c in played.values()), default=None)

        # chi deve ancora giocare dopo di me in questo turno
        pos = order.index(myposition)
        id_players_after = order[pos+1:] + order[:pos]
        id_players_after = [pid for pid in id_players_after if pid not in played]
        prese_rimaste_dopo = [int(preds[pid] - catches[pid]) for pid in id_players_after] if preds is not None else []
        check_necessità_prese_dopo = any(n > 0 for n in prese_rimaste_dopo)
        via_libera = all(n <= 0 for n in prese_rimaste_dopo) if id_players_after else True

        lowwin  = (lambda thr: next((c for c in H_ordinata if s(c) > thr), None))
        highnon = (lambda thr: next((c for c in reversed(H_ordinata) if s(c) <= thr), None))

        if necessità_presa:
            if opening:
                carta_giocata = H_ordinata[0] if check_necessità_prese_dopo else H_ordinata[-1]
            else:
                if via_libera:
                    prese_rimaste = int(self.prediction or 0) - int(catches[self.id])
                    if prese_rimaste == 1:
                        carte_vincenti = [c for c in H if s(c) > curmax]
                        carta_giocata = max(carte_vincenti, key=s) if carte_vincenti else H_ordinata[0]
                    else:
                        carta_giocata = lowwin(curmax) or H_ordinata[0]
                else:
                    carta_giocata = lowwin(curmax) or H_ordinata[0]
        else:
            if opening:
                carta_giocata = H_ordinata[0]
            else:
                carta_giocata = (highnon(curmax) if curmax is not None else H_ordinata[0]) or H_ordinata[0]

        H.remove(carta_giocata)
        return carta_giocata
