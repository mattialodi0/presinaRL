from colorama import init as colorama_init
from colorama import Fore
from colorama import Style
import random

suits = ['Bastoni', 'Spade', 'Coppe', 'Denari']
rank = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']

class Card:
    def __init__(self, suit, rank):
        self.suit = suit
        self.rank = rank

    def __str__(self):
        return f"{self.rank} di {self.suit}"
        
    def strc(self):
        if self.suit == 'Bastoni':
            return f"{self.rank} di {Fore.GREEN}{self.suit}{Style.RESET_ALL}"
        elif self.suit == 'Spade':
            return f"{self.rank} di {Fore.BLUE}{self.suit}{Style.RESET_ALL}"
        elif self.suit == 'Coppe':
            return f"{self.rank} di {Fore.RED}{self.suit}{Style.RESET_ALL}"
        elif self.suit == 'Denari':
            return f"{self.rank} di {Fore.YELLOW}{self.suit}{Style.RESET_ALL}"

    def __gt__(self, other):
        if suits.index(self.suit) > suits.index(other.suit):
            return True
        elif suits.index(self.suit) == suits.index(other.suit):
            return rank.index(self.rank) > rank.index(other.rank)
    
    def value(self):
        return suits.index(self.suit) * 10 + rank.index(self.rank)

class Deck:
    def __init__(self):
        self.cards = [Card(s, r) for s in suits for r in rank]
        random.shuffle(self.cards)
    
    def shuffle(self):
        random.shuffle(self.cards)

    def __str__(self):
        return ", ".join(str(card) for card in self.cards)
    
    def draw_card(self):
        return self.cards.pop() if self.cards else None

    def draw_hand(self, num_cards=5):
        hand = []
        for _ in range(num_cards):
            hand.append(self.draw_card())
        return hand
