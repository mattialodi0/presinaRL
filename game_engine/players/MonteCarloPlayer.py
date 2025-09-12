from random import choice, random
from time import time
from Deck import Deck
from players.RandomPlayer import RandomPlayer
from players.Player import Player
import sys
sys.path.append("..")


class MonteCarloPlayer(RandomPlayer):
    """Monte Carlo player class, all decisions are based on Monte Carlo simulations.
       A time limit is set for each decision."""

    def __init__(self, name="MonteCarloPlayer"):
        super().__init__(name)
        self.time_limit = 1  # seconds
        self.max_reps = 10000 # seconds

    # def make_prediction(self, game_state):
    #     self.prediction = 0
    #     game_state["hand"] = self.hand
    #     try:
    #         sim = simulateFromPrediction(game_state, self.time_limit, self.max_reps)
    #     except Exception as e:
    #         pass
    #     self.prediction = sim.run()
    #     return self.prediction

    def make_prediction_last_round(self, game_state):
        self.prediction = 0
        try:
            self.prediction = simulateFromLastPrediction(
                game_state, self.time_limit, self.max_reps)
        except Exception as e:
            print(e)
        return self.prediction

    # def play_card(self, game_state):
    #     self.card_played = None
    #     game_state["hand"] = self.hand
    #     try:
    #         sim = simulateFromPlay(game_state, self.time_limit, self.max_reps)
    #     except Exception as e:
    #         pass
    #     self.card_played = sim.run()
    #     self.hand.remove(self.card_played)
    #     return self.card_played


def simulateFromPrediction(game_state, hand, time_limit, max_reps):
    start_time = time()
    best_pred = 0
    possible_preds = [0, 1]
    preds_wins = {p: 0 for p in possible_preds}
    preds_losses = {p: 0 for p in possible_preds}
    rep = 0
    while True:
        if time() - start_time > time_limit or rep > max_reps:
            break

        pred = choice(possible_preds)
        players = {}
        d = Deck()
        for p in range(game_state["num_players"]):
            # nmake players
            players[p] = {}
            players[p]["hand"] = d.draw_card()
            # predictions
            players[p]["pred"] = choice([0, 1])

        vals = []
        for p in range(game_state["num_players"]):
            vals.append(players[p]["hand"].value())
        vals.append(hand[0].value())
        # determine winner
        max_val = max(vals)
        catchers = [i for i, v in enumerate(vals) if v == max_val]
        if pred == 0:
            if game_state["num_players"]-1 not in catchers:
                preds_wins[pred] += 1
            else:
                preds_losses[pred] += 1
        else:
            if game_state["num_players"]-1 in catchers:
                preds_wins[pred] += 1
            else:
                preds_losses[pred] += 1

        rep += 1
    return best_pred

def simulateFromLastPrediction(game_state, time_limit, max_reps):
    start_time = time()
    best_pred = 0
    possible_preds = [0, 1]
    preds_wins = {p: 0 for p in possible_preds}
    preds_losses = {p: 0 for p in possible_preds}
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
                preds_losses[pred] += 1
        else:
            if game_state["num_players"]-1 in catchers:
                preds_wins[pred] += 1
            else:
                preds_losses[pred] += 1

        rep += 1
    return best_pred
