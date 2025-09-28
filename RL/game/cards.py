class Card:
    def __init__(self, rank, suit):
        self.rank = rank
        self.suit = suit

    def __repr__(self):
        return f"{self.rank} of {self.suit}"

    def __lt__(self, other):
        return self.rank < other.rank

class Deck:
    def __init__(self):
        self.cards = self.create_deck()

    def create_deck(self):
        suits = ['Cups', 'Coins', 'Swords', 'Clubs']
        ranks = list(range(1, 11))  # Italian cards typically have ranks 1-10
        return [Card(rank, suit) for suit in suits for rank in ranks]

    def shuffle(self):
        import random
        random.shuffle(self.cards)

    def deal(self, n):
        return [self.cards.pop() for _ in range(n)] if len(self.cards) >= n else []

    def __repr__(self):
        return f"Deck of {len(self.cards)} cards"