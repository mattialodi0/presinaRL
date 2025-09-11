from Game import Game, PlayersGame, PlayableGame
from players.RandomPlayer import RandomPlayer
from players.CiroPlayer import CiroPlayer
from players.LodoPlayer import LodoPlayer
from itertools import permutations


if __name__ == "__main__":

    # --- simple game ---
    # game = Game(r=2)
    # game.play(verbose=True)

    # --- play as human ---
    # game = PlayableGame(r=1)
    # game.play(verbose=True)


    # --- alg. player matchup ---
    ciroPlayer_wins = 0
    lodoPlayer_wins = 0
    ciroPlayer_errs = 0
    lodoPlayer_errs = 0
    for _ in range(5): # repeat to reduce variance
        # all possible permutations
        perm = permutations(['C','C','L','L'], 4)
        for p in perm:
            players = []
            for i,s in enumerate(p):
                if s == 'C':
                    players.append(CiroPlayer(i))
                else:
                    players.append(LodoPlayer(i))

            game = PlayersGame(players, r=1000)
            game.play(verbose=False)
            stats = game.return_stats()
            for i, w in enumerate(stats["winners"]):
                if w == 1:
                    if p[i] == 'C':
                        ciroPlayer_wins += 1
                    else:
                        lodoPlayer_wins += 1
            for i, e in enumerate(stats["tot_errors"]):
                if p[i] == 'C':
                    ciroPlayer_errs += e
                else:
                    lodoPlayer_errs += e

    print("CiroPlayer wins:", ciroPlayer_wins)
    print("LodoPlayer wins:", lodoPlayer_wins)
    print("CiroPlayer errors:", ciroPlayer_errs)
    print("LodoPlayer errors:", lodoPlayer_errs)
