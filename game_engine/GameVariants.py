from Game import Game
from players.HumanPlayer import HumanPlayer
from PlayerNode import link_players

    
class PlayersGame(Game):
    def __init__(self, players, r=5, e=-1, s=0):
        n = len(players)
        if n > 8:
            raise ValueError("Too many players")
        
        super().__init__(n, r, e, s)
        self.players = players
        self.current_player = link_players(self.players)

class PlayableGame(Game):
    def __init__(self, n=5, r=5, e=-1, s=0, human_pos=0):
        super().__init__(n, r, e, s)
        if human_pos < 0 or human_pos >= n:
            raise ValueError("Invalid human player position")
        self.human_pos = human_pos
        self.players[human_pos] = HumanPlayer(id=human_pos)
        self.current_player = link_players(self.players)
    
    def play(self):
        super().play(2)