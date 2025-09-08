from Deck import Deck
from Deck import Card
from Player import Player
import numpy as np


class Game:
    def __init__(self, n=5, r=5, e=3):
        self.deck = Deck()
        self.round = 0
        self.max_rounds = r
        self.players = [Player() for _ in range(n)]
        self.errors = np.zeros((r, n))
        self.predictions = np.zeros((r, n))
        self.catches = np.zeros((r, n))
        self.max_errors = e
        self.current_player_index = 0

    def play(self):
        self.deck.shuffle()
        for round in range(self.max_rounds):
            # deal hand
            for player in self.players:
                player.take_hand(self.deck.draw_hand())

            # make prediction
            for _ in range(len(self.players)):
                c = self.current_player_index
                i = 1
                player = self.players[i]
                last = i == self.max_players
                self.predictions[round, c] = player.make_prediction(self.predictions[round].tolist(), last, round)
                c += 1
                i += 1

                # play cards
                for turn in range(5-round):
                    for _ in range(len(self.players)):
                        c = self.current_player_index
                        played_cards = []
                        player = self.players[c]
                        played_cards.append(player.play_card(self.played_cards))
                        c += 1
                    
                    if not len(played_cards) == self.max_players:
                        raise ValueError("Not all players played a card")
                    max = None
                    max_idx = None
                    for idx, card in enumerate(played_cards):
                        if max is None or card > max:
                            max = card
                            max_idx = idx
                    self.catches[round, max_idx] += 1

            # check errors
            for i, player in enumerate(self.players):
                self.errors[round, i] = abs(self.predictions[round, i] - self.catches[round, i])
                if np.sum(self.errors[:, i]) > self.max_errors:
                    print(f"Player {i} has exceeded the maximum number of errors.")
                    return

            self.current_player_index = (self.current_player_index + 1) % len(self.players)

        print("Game Over")