from random import choice
from time import time
from Deck import Deck
from players.RandomPlayer import RandomPlayer
import numpy as np
import sys
sys.path.append("..")


class MCPlayerLite(RandomPlayer):

    def __init__(self, name="MCPlayerLite"):
        super().__init__(name)
        self.time_limit = 0.05   # tempo max per decisione (secondi)
        self.max_reps   = 500    # simulazioni max per decisione

    # ------------------- PREDIZIONE (mano > 1 carta) -------------------
    def make_prediction(self, game_state):
        start_time   = time()
        num_players  = game_state["num_players"]
        hand_size    = game_state["hand_size"]
        order        = game_state["players_position"]
        my_pos       = order.index(self.id)

        possible_preds = list(range(hand_size + 1))
        preds_wins     = {p: 0 for p in possible_preds}

        for rep in range(self.max_reps):
            if time() - start_time > self.time_limit:
                break

            d = Deck()
            # togli le mie carte dal mazzo simulato
            for c in self.hand:
                d.remove(c)

            # scegli una mia predizione candidata
            my_pred = choice(possible_preds)

            # costruisci i giocatori simulati
            players = {}
            for p in range(num_players):
                players[p] = {}
                if p < my_pos:
                    players[p]["hand"] = d.draw_hand(hand_size)
                    # usa la predizione già fatta (o 0 se None)
                    prev = game_state["predictions_made"][p]
                    players[p]["pred"] = int(prev) if prev is not None else 0
                elif p == my_pos:
                    players[p]["hand"] = self.hand.copy()
                    players[p]["pred"] = my_pred
                else:
                    players[p]["hand"] = d.draw_hand(hand_size)
                    players[p]["pred"] = choice(possible_preds)

            # se nell'ordine simulato io sono l'ultimo, applica la regola "somma != hand_size"
            if my_pos == num_players - 1 and hand_size > 1:
                total = sum(int(players[q]["pred"]) for q in range(num_players))
                if total == hand_size:
                    # micro-aggiustamento
                    if players[my_pos]["pred"] > 0:
                        players[my_pos]["pred"] -= 1
                    else:
                        players[my_pos]["pred"] += 1

            # simula il round (scelta casuale delle carte)
            for _ in range(hand_size):
                played = []
                for p in range(num_players):
                    card = choice(players[p]["hand"])
                    players[p]["hand"].remove(card)
                    played.append(card)
                vals = [c.value() for c in played]
                winner = vals.index(max(vals))
                players[winner]["catches"] = players[winner].get("catches", 0) + 1

            if players[my_pos]["pred"] == players[my_pos].get("catches", 0):
                preds_wins[players[my_pos]["pred"]] += 1

        # scelta migliore
        self.prediction = max(preds_wins, key=preds_wins.get)

        # 🔧 correzione extra (sul valore finale): se sono davvero ultimo nel gioco corrente
        preds_made = game_state["predictions_made"]
        if order[-1] == self.id and hand_size > 1:
            safe_sum = sum(int(x) for x in preds_made if x is not None)
            if safe_sum + int(self.prediction) == hand_size:
                self.prediction = int(self.prediction) - 1 if self.prediction > 0 else int(self.prediction) + 1

        return int(self.prediction)

    # ------------------- PREDIZIONE (mano = 1 carta) -------------------
    def make_prediction_last_round(self, game_state):
        start_time = time()
        wins = {0: 0, 1: 0}

        # carte visibili degli altri
        others = [c.value() for c in game_state["cards_visible"]]
        # possibili valori rimasti (inclusa la mia)
        remaining = [v for v in range(40) if v not in others]

        for rep in range(self.max_reps):
            if time() - start_time > self.time_limit:
                break
            # estrai casualmente la mia carta
            my_val = choice(remaining)
            vals   = others + [my_val]
            # chi vince la singola presa?
            winner_idx = vals.index(max(vals))
            # assumo che io sia l'ultimo della lista "vals"
            wins[1 if winner_idx == len(vals) - 1 else 0] += 1

        self.prediction = max(wins, key=wins.get)
        return int(self.prediction)

    # ------------------- SCELTA CARTA -------------------
    def play_card(self, game_state):
        start_time   = time()
        order        = game_state["players_position"]
        my_pos       = order.index(self.id)
        num_players  = game_state["num_players"]
        to_play      = game_state["hand_size"] - game_state["turn"]

        my_vals  = [c.value() for c in self.hand]
        scores   = {v: 0 for v in my_vals}

        for rep in range(self.max_reps):
            if time() - start_time > self.time_limit:
                break

            # scegli una mia carta candidata
            cand = choice(my_vals)

            # mazzo simulato: togli le mie carte e (se serve) le carte già giocate nel turno
            d = Deck()
            for c in self.hand:
                d.remove(c)
            if game_state["turn"] > 0:
                for c in game_state["played_cards"].values():
                    d.remove(c)

            # costruisci mani fittizie per gli altri per il resto del round
            players = {}
            for p in range(num_players):
                players[p] = {}
                if p == my_pos:
                    # la mia mano "residua" (tolgo la candidata dalla lista di valori)
                    players[p]["hand"] = self.hand.copy()
                else:
                    players[p]["hand"] = d.draw_hand(to_play)

            # simula le ultime 'to_play' prese; nella prima io gioco la carta candidata
            catches = [0] * num_players
            for t in range(to_play):
                played = []
                for p in range(num_players):
                    if p == my_pos and t == 0:
                        # gioco la carta candidata
                        # (trova l'oggetto Card corrispondente al valore "cand")
                        idx = next(i for i, cc in enumerate(players[p]["hand"]) if cc.value() == cand)
                        card = players[p]["hand"].pop(idx)
                    else:
                        card = choice(players[p]["hand"])
                        players[p]["hand"].remove(card)
                    played.append(card)
                vals = [c.value() for c in played]
                winner = vals.index(max(vals))
                catches[winner] += 1

            # punto se alla fine rispetto la mia predizione
            if catches[my_pos] == int(self.prediction or 0):
                scores[cand] += 1

        # scegli la carta con score migliore e rimuovila dalla mano reale
        best_val = max(scores, key=scores.get)
        best_idx = next(i for i, c in enumerate(self.hand) if c.value() == best_val)
        return self.hand.pop(best_idx)
