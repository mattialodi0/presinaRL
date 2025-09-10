from Game import Game, PlayersGame, PlayableGame
from players.RandomPlayer import RandomPlayer
from players.Player import Player

if __name__ == "__main__":
    # game = Game(r=2)
    # game = PlayableGame(r=1)
    # game = PlayableGame(r=1)
    game = PlayersGame([RandomPlayer(i) for i in range(2)], r=2)
    game.play(verbose=True)
    
    # stats = game.return_stats()
    # print(stats)
