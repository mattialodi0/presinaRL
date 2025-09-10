from Game import Game, PlayersGame, PlayableGame
from players.RandomPlayer import RandomPlayer
from players.CiroPlayer import CiroPlayer

if __name__ == "__main__":
    # game = Game(r=2)
    # game = PlayableGame(r=1)
    # game = PlayableGame(r=1)
    game = PlayersGame([RandomPlayer(0), RandomPlayer(1), RandomPlayer(2), CiroPlayer(3)], r=10)
    game.play(verbose=False)
    
    stats = game.return_stats()
    print(stats)
