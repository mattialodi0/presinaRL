from Game import Game
from GameVariants import PlayersGame, PlayableGame
from players.CiroPlayer import CiroPlayer
from players.MCPlayerLite import MCPlayerLite
from players.MonteCarloPlayer import MonteCarloPlayer
from players.LodoPlayer1 import LodoPlayer1
from itertools import permutations
import time

start = time.time()

# --- alg. player matchup ---
ciroPlayer_wins = 0
lodo_wins = 0
ciroPlayer_errs = 0
lodo_errs = 0

for _ in range(5):  # repeat to reduce variance
    # all possible permutations
    perm = permutations(['C', 'C', 'L', 'L'], 4)
    for p in perm:
        players = []
        for i, s in enumerate(p):
            if s == 'C':
                players.append(CiroPlayer(i))
            else:
                players.append(LodoPlayer1(i))

        game = PlayersGame(players, r=1, s=4)
        game.play(verbose=False)
        stats = game.return_stats()
        for i, w in enumerate(stats["winners"]):
            if w == 1:
                if p[i] == 'C':
                    ciroPlayer_wins += 1
                else:
                    lodo_wins += 1
        for i, e in enumerate(stats["tot_errors"]):
            if p[i] == 'C':
                ciroPlayer_errs += e
            else:
                lodo_errs += e

print("CiroPlayer wins:", ciroPlayer_wins)
print("Lodo wins:", lodo_wins)
print("CiroPlayer errors:", ciroPlayer_errs)
print("Lodo errors:", lodo_errs)

end = time.time()
print(f"Tempo di esecuzione: {end - start:.6f} secondi")