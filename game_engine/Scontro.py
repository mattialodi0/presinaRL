from Game import Game
from GameVariants import PlayersGame, PlayableGame
from players.CiroPlayer import CiroPlayer
from players.MCPlayerLite import MCPlayerLite
from players.MonteCarloPlayer import MonteCarloPlayer
from itertools import permutations
import time

start = time.time()

# --- alg. player matchup ---
ciroPlayer_wins = 0
MCPlayer_wins = 0
ciroPlayer_errs = 0
MCPlayer_errs = 0

for _ in range(5):  # repeat to reduce variance
    # all possible permutations
    perm = permutations(['C', 'C', 'M', 'M'], 4)
    for p in perm:
        players = []
        for i, s in enumerate(p):
            if s == 'C':
                players.append(CiroPlayer(i))
            else:
                players.append(MonteCarloPlayer(i))

        game = PlayersGame(players, r=5)
        game.play(verbose=False)
        stats = game.return_stats()
        for i, w in enumerate(stats["winners"]):
            if w == 1:
                if p[i] == 'C':
                    ciroPlayer_wins += 1
                else:
                    MCPlayer_wins += 1
        for i, e in enumerate(stats["tot_errors"]):
            if p[i] == 'C':
                ciroPlayer_errs += e
            else:
                MCPlayer_errs += e

print("CiroPlayer wins:", ciroPlayer_wins)
print("MCPlayer wins:", MCPlayer_wins)
print("CiroPlayer errors:", ciroPlayer_errs)
print("MCPlayer errors:", MCPlayer_errs)

end = time.time()
print(f"Tempo di esecuzione: {end - start:.6f} secondi")