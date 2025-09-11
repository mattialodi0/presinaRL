
class Player:
    """Base class for a player in the game."""
    def __init__(self, id=0):
        self.id = id
        self.hand = []
        self.prediction = 0
        self.round_errors = 0

    def take_hand(self, hand):
        self.hand = hand
        return self.hand

    def make_prediction(self, game_state):
        """ Here game_state contains:
            - num of players left -> int
            - hand size -> int
            - self position in the round -> int
            - predictions made so far -> list
        """
        last = game_state["self_position"] == game_state["num_players"] - 1
        if last:
            if sum(game_state["predictions_made"]) == game_state["hand_size"]:
                self.prediction = 1
            else:
                self.prediction = 0
        else:
                self.prediction = 0
        return self.prediction
    
    def make_prediction_last_round(self, game_state):
        """ Here game_state contains:
            - num of players left -> int
            - cards visible -> list
            - players (id) position in the round -> [int]
            - predictions made so far -> [int]
        """
        self.prediction = 0
        return self.prediction

    def play_card(self, game_state):
        """ Here game_state contains:
            - num of players left -> int
            - hand_size -> int
            - players (id) position in the round -> [int]
            - predictions made -> [int]
            - errors made so far -> [int]
            - catches made so far -> [int]
            - cards played so far -> [int]
        """
        return self.hand.pop() if self.hand else None