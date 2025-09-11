from Deck import Deck, Card
from players.Player import Player
from players.HumanPlayer import HumanPlayer
from players.RandomPlayer import RandomPlayer
from PlayerNode import PlayerNode, link_players, remove_player, goto_player
from colorama import init as colorama_init
from colorama import Fore
from colorama import Style
import numpy as np
import math


class Game:
    def __init__(self, n=5, r=5, e=-1, s=0):
        """Initialize the game with given parameters.
            - n: number of players (max 8)
            - r: number of rounds
            - e: max errors allowed (-1 for infinite)
            - s: starting round 
        """

        if n > 8:
            raise ValueError("Too many players")
        if s > 5:
            raise ValueError("Too many starting cards")

        colorama_init()
        self.deck = Deck()
        self.round = 0
        self.max_rounds = r
        self.starting_round = s
        self.max_players = n
        self.players = [RandomPlayer(id) for id in range(n)]
        self.current_player = link_players(self.players)
        self.player_node = self.current_player

        self.errors = np.zeros((r, n))
        self.predictions = np.zeros((r, n))
        self.catches = np.zeros((r, n))
        self.first_in_turn = np.zeros((r, n))
        self.last_in_turn = np.zeros((r, n))
        self.max_errors = e
        if e == -1:
            self.max_errors = math.inf

    def play(self, verbose=2):
        """ verbose: 0 no output, 1 only game stats, 2 output for human interaction, 3 full output"""
        self.verbose = verbose
        for round in range(self.max_rounds):

            hand_size = 5 - ((round+self.starting_round) % 5)

            if self.verbose >= 2:
                print(f"\n--- Round {round+1}: {hand_size} card(s) ---")

            if hand_size > 1:
                self.dealHands(hand_size)
                self.makePredictions(round, hand_size)

                # play cards
                for turn in range(hand_size):
                    self.first_in_turn[round, self.player_node.player.id] += 1
                    self.last_in_turn[round, goto_player(
                        self.player_node, (self.player_node.player.id+len(self.players)-1) % len(self.players)).player.id] += 1
                    if self.verbose >= 2:
                        print(f"--- Turn {turn+1} ---")

                    played_cards = self.playCards()
                    self.determineCatches(round, played_cards)
            else:
                visible_cards = []
                for p in self.players:
                    hand = self.deck.draw_hand(1)[0]
                    visible_cards.append(hand)
                    if self.verbose >= 3:
                        print(f"Player {p.id} hand: [{hand.strc()}]")

                self.makePredictions(round, hand_size, visible_cards)
                played_cards = {}
                for p in self.players:
                    played_cards[p.id] = visible_cards[p.id]
                self.determineCatches(round, played_cards)

                if self.verbose == 2:
                    print(f"All cards: [{', '.join([card.strc() for card in visible_cards])}]")

            if self.verbose >= 1:
                print(" ")

            removed_players = self.checkForErrors(round)
            if removed_players:
                if len(self.players) == 1 and self.verbose >= 1:
                    print(f"Winner: {self.players[0].id}")
                if len(self.players) <= 1:
                    if self.verbose >= 1:
                        print("Game Over")
                    break

            self.current_player = self.current_player.next
            self.deck = Deck()

        if verbose >= 1:
            print("Game Over")
            print(
                f"Players remaining: {[player.id for player in self.players]}")

    def dealHands(self, hand_size):
        for player in self.players:
            hand = self.deck.draw_hand(hand_size)
            player.take_hand(hand)
            if self.verbose >= 3:
                card_strs = hand[0].strc()
                for card in hand[1:]:
                    card_strs += ", " + card.strc()
                print(f"Player {player.id} hand: [{card_strs}]")

    def makePredictions(self, round, hand_size, visible_cards=[]):
        current_preds = []
        self.player_node = self.current_player
        game_state = {
            "num_players": len(self.players),
            "hand_size": hand_size,
            "players_position": [],
            "predictions_made": []
        }
        t = self.player_node
        for _ in self.players:
            game_state["players_position"].append(t.player.id)
            t = t.next
        for i in range(len(self.players)):
            game_state["predictions_made"] = current_preds

            if hand_size > 1:
                p = self.player_node.player.make_prediction(game_state)
                current_preds.append(p)
                self.predictions[round, self.player_node.player.id] = p
            else:
                c = visible_cards.copy()
                c.pop(i)
                game_state["cards_visible"] = c
                p = self.player_node.player.make_prediction_last_round(game_state)
                current_preds.append(p)
                self.predictions[round, self.player_node.player.id] = p

            if self.verbose >= 2:
                print(f"Player {self.player_node.player.id}: prediction: {p}")
            self.player_node = self.player_node.next
        
        if hand_size > 1 and np.sum(self.predictions[round]) == hand_size:
            raise ValueError(f"Invalid prediction from player {self.player_node.player.id}")
        self.player_node = self.current_player

    def playCards(self):
        played_cards = {}
        for _ in range(len(self.players)):
            card = self.player_node.player.play_card(
                list(map(lambda c: str(c), played_cards.values())))
            if self.verbose:
                print(
                    f"Player {self.player_node.player.id} plays: {card.strc()}")
            played_cards[self.player_node.player.id] = card
            self.player_node = self.player_node.next

        if not len(played_cards.values()) == len(self.players):
            raise ValueError("Not all players played a card")

        return played_cards

    def determineCatches(self, round, played_cards):
        ace = False
        for id, card in played_cards.items():
            if card.rank == '1' and card.suit == 'Denari':
                if self.predictions[round, id] > self.catches[round, id]:
                    ace = True
                    self.catches[round, id] += 1
                    self.player_node = goto_player(self.player_node, id)
                    if self.verbose >= 2:
                        print(f"Player {id} takes")
                else:
                    # lowest possible card
                    played_cards[id] = Card('Bastoni', '1')

        if not ace:
            max = None
            max_idx = None
            for idx, card in played_cards.items():
                if max is None or card > max:
                    max = card
                    max_idx = idx
            self.catches[round, max_idx] += 1
            self.player_node = goto_player(self.player_node, max_idx)
            if self.verbose >= 2:
                print(f"Player {max_idx} takes")

    def checkForErrors(self, round):
        removed_players = []
        for i, player in enumerate(self.players):
            e = abs(self.predictions[round, i] - self.catches[round, i])
            if e > 0:
                self.errors[round, i] = e
                if self.verbose >= 2:
                    print(
                        f"{Fore.MAGENTA}Player {player.id} made {int(e)} error(s){Style.RESET_ALL}")
            if np.sum(self.errors[:, player.id]) > self.max_errors:
                # delete player
                if self.verbose >= 2:
                    print(f"Player {player.id} is eliminated.")
                p = self.players.pop(i)
                remove_player(self.current_player, p.id)
                removed_players.append(p.id)

        return removed_players

    def return_stats(self):
        winners = np.zeros(self.max_players)
        if self.max_errors == math.inf:
            min = np.min(np.sum(self.errors, axis=0))
            for i, e in enumerate(np.sum(self.errors, axis=0)):
                if e == min:
                    winners[i] = 1
        else:
            for p in self.players:
                winners[p.id] = 1

        stats = {
            "predictions": self.predictions,
            "catches": self.catches,
            "errors": self.errors,
            "first_in_turn": self.first_in_turn,
            "last_in_turn": self.last_in_turn,
            "tot_predictions": np.sum(self.predictions, axis=0),
            "tot_catches": np.sum(self.catches, axis=0),
            "tot_errors": np.sum(self.errors, axis=0),
            "tot_first": np.sum(self.first_in_turn, axis=0),
            "tot_last": np.sum(self.last_in_turn, axis=0),
            "winners": winners,
        }
        return stats
