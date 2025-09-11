from players.Player import Player


class HumanPlayer(Player):
    """Human player class, all decisions are made via user input."""
    def __init__(self, id=0):
        super().__init__(id)
        print(f"Registered as Player {id}")

    def take_hand(self, hand):
        super().take_hand(sorted(hand, key=lambda card: (card.suit, card.rank)))

    def make_prediction(self, game_state):
        last = game_state["players_position"].index(self.id) == game_state["num_players"] - 1
        card_strs = self.hand[0].strc()
        for card in self.hand[1:]:
            card_strs += ", " + card.strc()
        print(f"Your hand: [{card_strs}]")
        # if game_state["predictions_made"]:
        #     print(f"Predictions made so far: {game_state['predictions_made']} (sum: {sum(game_state['predictions_made'])})")
        while True:
            try:
                pred = int(input(f"Enter your prediction (0 to {game_state['hand_size']}): "))
                if 0 <= pred <= game_state["hand_size"]:
                    if last and (sum(game_state["predictions_made"]) + pred == game_state["hand_size"]):
                        print("Invalid prediction: cannot make the sum of predictions equal to hand size.")
                    else:
                        self.prediction = pred
                        return self.prediction
                else:
                    print(f"Prediction must be between 0 and {game_state['hand_size']}.")
            except ValueError:
                print("Invalid input. Please enter an integer.")
    
    def make_prediction_last_round(self, game_state): # random
        cards = game_state["cards_visible"]
        card_strs = cards[0].strc()
        for card in cards[1:]:
            card_strs += ", " + card.strc()
        print(f"The cards you see: [{card_strs}]")
        # if game_state["predictions_made"]:
        #     print(f"Predictions made so far: {game_state['predictions_made']} (sum: {sum(game_state['predictions_made'])})")
        while True:
            try:
                pred = int(input(f"Enter your prediction (0 or 1): "))
                if pred not in [0, 1]:
                    print(f"Prediction must be 0 or 1.")
                else:
                    self.prediction = pred
                    return self.prediction
            except ValueError:
                print("Invalid input. Please enter an integer.")
        

    def play_card(self, game_state):
        card_strs = "0: " + self.hand[0].strc()
        i = 1
        for card in self.hand[1:]:
            card_strs += f", {i}: " + card.strc()
            i += 1
        print(f"Your hand: [{card_strs}]")

        while True:
            try:
                card_index = int(input(f"Enter the index of the card to play (0 to {len(self.hand)-1}): "))
                if 0 <= card_index < len(self.hand):
                    return self.hand.pop(card_index)
                else:
                    print(f"Index must be between 0 and {len(self.hand)-1}.")
            except ValueError:
                print("Invalid input. Please enter an integer.")