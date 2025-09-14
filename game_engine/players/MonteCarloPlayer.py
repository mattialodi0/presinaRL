from random import choice, random
from time import time
from Deck import Deck, Card
from players.RandomPlayer import RandomPlayer
from players.Player import Player
import numpy as np
import sys
sys.path.append("..")


class MonteCarloPlayer(RandomPlayer):
    """Monte Carlo player class, all decisions are based on Monte Carlo simulations.
       A time limit is set for each decision."""

    def __init__(self, name="MonteCarloPlayer"):
        super().__init__(name)
        self.time_limit = 1  # seconds
        self.max_reps = 10000 # seconds

    def make_prediction(self, game_state):
        self.prediction = 0
        self.prediction = self.__simulateFromPrediction(game_state, self.time_limit, self.max_reps)
        return self.prediction
            

    def make_prediction_last_round(self, game_state):
        self.prediction = 0
        self.prediction = self.__simulateFromLastPrediction(game_state, self.time_limit, self.max_reps)
        return self.prediction
        
    def play_card(self, game_state):
        self.card_played = None
        card_value = self.__simulateFromPlay(game_state, self.time_limit, self.max_reps)
        vs = [c.value() for c in self.hand]
        self.card_played = self.hand[vs.index(card_value)]
        self.hand.pop(vs.index(card_value))
        return self.card_played
        

    def __simulateFromPrediction(self, game_state, time_limit, max_reps):
        start_time = time()
        num_players = game_state["num_players"]
        hand_size = game_state["hand_size"]
        players_position = game_state["players_position"]
        player_pos = players_position.index(self.id)
        best_pred = 0
        possible_preds = list(range(hand_size+1))
        preds_wins = {p: 0 for p in possible_preds}
        rep = 0
        while True:
            if time() - start_time > time_limit or rep > max_reps:
                break
            
            d  = Deck()
            pred = choice(possible_preds)
            for c in self.hand:
                d.remove(c)

            players = {}
            for p in range(num_players):
                if p < player_pos:
                    players[p] = {}
                    players[p]["hand"] = d.draw_hand(hand_size)
                    players[p]["pred"] = game_state["predictions_made"][p] if game_state["predictions_made"][p] is not None else 0
                elif p == player_pos:
                    players[p] = {}
                    players[p]["hand"] = self.hand.copy()
                    players[p]["pred"] = pred
                    while p == num_players-1:
                        if not sum(players[q]["pred"] for q in range(num_players)) == hand_size:
                            break
                        pred = choice(possible_preds)
                        players[p]["pred"] = pred
                else:
                    players[p] = {}
                    players[p]["hand"] = d.draw_hand(hand_size)
                    players[p]["pred"] = choice(possible_preds)
                    while p == num_players-1:
                        if not sum(players[q]["pred"] for q in range(num_players)) == hand_size:
                            break
                        players[p]["pred"] = choice(possible_preds)
            

            # simulate rest of round
            for t in range(hand_size):
                played_cards = []
                for p in range(num_players):
                    card = choice(players[p]["hand"])
                    played_cards.append(card)
                    players[p]["hand"].remove(card)
                    if p == player_pos and t == 0:
                        pred = players[player_pos]["pred"]
                # determine winner
                vals = [c.value() for c in played_cards]
                max_val = max(vals)
                catcher = vals.index(max_val)
                if "catches" in players[catcher]:
                    players[catcher]["catches"] += 1
                else:
                    players[catcher]["catches"] = 1

            # check if the prediction was correct, given the card played
            if players[player_pos]["pred"] == players[player_pos].get("catches", 0):
                preds_wins[pred] += 1

            rep += 1 

        best_pred = max(preds_wins, key=preds_wins.get)
        return best_pred

    def __simulateFromLastPrediction(self, game_state, time_limit, max_reps):
        start_time = time()
        best_pred = 0
        possible_preds = [0, 1]
        preds_wins = {p: 0 for p in possible_preds}
        rep = 0
        while True:
            if time() - start_time > time_limit or rep > max_reps:
                break

            pred = choice(possible_preds)
            players = {}
            for p in range(game_state["num_players"]-1):
                # make players
                players[p] = {}
                players[p]["hand"] = game_state["cards_visible"][p]
                # predictions
                players[p]["pred"] = choice([0, 1])
            
            vals = []
            for p in range(game_state["num_players"]-1):
                vals.append(players[p]["hand"].value())
            vals.append(choice([v for v in range(40) if v not in vals]))
            # determine winner
            max_val = max(vals)
            catchers = [i for i, v in enumerate(vals) if v == max_val]
            if pred == 0:
                if game_state["num_players"]-1 not in catchers:
                    preds_wins[pred] += 1
            else:
                if game_state["num_players"]-1 in catchers:
                    preds_wins[pred] += 1

            rep += 1 

        best_pred = max(preds_wins, key=preds_wins.get)
        return best_pred

    def __simulateFromPlay(self, game_state, time_limit, max_reps):
        start_time = time()
        best_card = 0
        card_wins = {c.value(): 0 for c in self.hand}
        num_players = game_state["num_players"]
        predictions_left = np.abs(game_state["predictions"]-game_state["catches"])
        players_position = game_state["players_position"]
        hand_size = game_state["hand_size"]-game_state["turn"]
        player_pos = -1

        rep = 0
        while True:
            if time() - start_time > time_limit or rep > max_reps:
                break
            
            d = Deck()
            # if turn > 0 remove played cards from deck
            if game_state["turn"] > 0:
                for c in game_state["played_cards"].values():
                    d.remove(c)
            for c in self.hand:
                d.remove(c)


            chosen_card = None
            # make player based on predictions and already played cards
            players = {}
            for p in range(num_players):
                if players_position[p] == self.id:
                    players[p] = {}
                    players[p]["hand"] = self.hand.copy()
                    players[p]["pred"] = self.prediction
                    player_pos = p
                else:
                    players[p] = {}
                    players[p]["hand"] = d.draw_hand(hand_size)
                    players[p]["pred"] = predictions_left[p] if predictions_left[p] > 0 else 0
            
            # simulate rest of round
            for t in range(hand_size):
                played_cards = []
                for p in range(num_players):
                    card = choice(players[p]["hand"])
                    played_cards.append(card)
                    players[p]["hand"].remove(card)
                    if p == player_pos and t == 0:
                        chosen_card = card
                # determine winner
                vals = [c.value() for c in played_cards]
                max_val = max(vals)
                catcher = vals.index(max_val)
                if "catches" in players[catcher]:
                    players[catcher]["catches"] += 1
                else:
                    players[catcher]["catches"] = 1

            # check if the prediction was correct, given the card played
            if players[player_pos]["pred"] == players[player_pos].get("catches", 0):
                card_wins[chosen_card.value()] += 1

            rep += 1

        best_card = max(card_wins, key=card_wins.get)
        return best_card