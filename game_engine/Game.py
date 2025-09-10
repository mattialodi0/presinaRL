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
        if n > 8:
            raise ValueError("Too many players")
        if s > 5:
            raise ValueError("Too many starting cards")
        
        self.deck = Deck()
        self.round = 0
        self.max_rounds = r
        self.starting_round = s
        self.max_players = n
        self.players = [RandomPlayer(id) for id in range(n)]
        self.current_player = link_players(self.players)
        self.errors = np.zeros((r, n))
        self.predictions = np.zeros((r, n))
        self.catches = np.zeros((r, n))
        self.first_in_turn = np.zeros((r, n))
        self.last_in_turn = np.zeros((r, n))
        self.max_errors = e
        if e == -1:
            self.max_errors = math.inf

    def play(self, verbose=True, show_hands=True):
        for round in range(self.max_rounds):
            # hand_size = 5 - (round % 5)
            hand_size = 5 - ((round+self.starting_round) % 5)
            
            if verbose:
                print(f"\n--- Round {round+1}: {hand_size} card(s) ---")
            
            # deal hand
            for player in self.players:
                hand = self.deck.draw_hand(hand_size)
                player.take_hand(hand)
                card_strs = hand[0].strc()
                for card in hand[1:]:
                    card_strs += ", " + card.strc()
                if verbose and show_hands:
                    print(f"Player {player.id} hand: [{card_strs}]")

            # make prediction
            current_preds = []
            player_node = self.current_player
            for _ in range(len(self.players)):
                if player_node.next.player.id == self.current_player.player.id:
                    last = True
                else:
                    last = False
                p = player_node.player.make_prediction(current_preds, last, hand_size)
                current_preds.append(p)
                self.predictions[round, player_node.player.id] = p
                if verbose:
                    print(f"Player {player_node.player.id} prediction: {p}")
                player_node = player_node.next
                # TODO check last prediction
            player_node = self.current_player

            # play cards
            for turn in range(hand_size):
                self.first_in_turn[round, player_node.player.id] += 1
                self.last_in_turn[round, goto_player(player_node, (player_node.player.id+len(self.players)-1)%len(self.players)).player.id] += 1
                if verbose:
                    print(f"--- Turn {turn+1} ---")
                played_cards = {}
                for _ in range(len(self.players)):
                    card = player_node.player.play_card(list(map(lambda c: str(c), played_cards.values())))
                    if verbose:
                        print(f"Player {player_node.player.id} plays: {card.strc()}")
                    played_cards[player_node.player.id] = card
                    player_node = player_node.next

                if not len(played_cards.values()) == len(self.players):
                    raise ValueError("Not all players played a card")
                
                # determine catches
                ace = False
                for id, card in played_cards.items():
                    if card.rank == '1' and card.suit == 'Denari':
                        if self.predictions[round, id] > self.catches[round, id]:
                            ace = True
                            self.catches[round, id] += 1
                            player_node = goto_player(player_node, id)
                            if verbose:
                                print(f"Player {id} takes")
                        else:
                            played_cards[id] = Card('Bastoni', '1')  # lowest possible card

                if not ace:
                    max = None
                    max_idx = None
                    for idx, card in played_cards.items():
                        if max is None or card > max:
                            max = card
                            max_idx = idx
                    self.catches[round, max_idx] += 1
                    player_node = goto_player(player_node, max_idx)
                    if verbose:
                        print(f"Player {max_idx} takes")

            if verbose:
                print(" ")

            # check errors
            removed_players = []
            for i, player in enumerate(self.players):
                e = abs(self.predictions[round, i] - self.catches[round, i])
                if e > 0:
                    self.errors[round, i] = e
                    if verbose:
                        print(f"{Fore.MAGENTA}Player {player.id} made {int(e)} error(s){Style.RESET_ALL}")
                if np.sum(self.errors[:, player.id]) > self.max_errors:
                    # delete player
                    if verbose:
                        print(f"Player {player.id} is eliminated.")
                    p = self.players.pop(i)
                    remove_player(self.current_player, p.id)
                    removed_players.append(p.id)
            
            if removed_players:
                if len(self.players) < 1:
                    print("Game Over")
                    print("Draw")
                    return
                if len(self.players) == 1:
                    print("Game Over")
                    print(f"Winner: {self.players[0].id}")
                    return

            self.current_player = self.current_player.next
            self.deck = Deck()

        if verbose:
            print("Game Over")
            print(f"Players remaining: {[player.id for player in self.players]}")

    def return_stats(self):
        winners = np.zeros(self.max_players)
        if self.max_errors == math.inf:
            min = np.min(self.errors[-1])
            for i, e in enumerate(self.errors[-1]):
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
    
class PlayersGame(Game):
    def __init__(self, players, r=5, e=-1, s=0):
        n = len(players)
        if n > 8:
            raise ValueError("Too many players")
        
        super().__init__(n, r, e, s)
        self.players = players
        self.current_player = link_players(self.players)

class PlayableGame(Game):
    def __init__(self, n=5, r=5, e=-1, human_pos=0):
        super().__init__(n, r, e)
        if human_pos < 0 or human_pos >= n:
            raise ValueError("Invalid human player position")
        self.human_pos = human_pos
        self.players[human_pos] = HumanPlayer(id=human_pos)
        self.current_player = link_players(self.players)
    
    def play(self):
        super().play(True, show_hands=False)