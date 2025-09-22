from Game import Game
from GameVariants import PlayersGame, PlayableGame
from players.CiroPlayer import CiroPlayer
from players.LodoPlayer1 import LodoPlayer1
import time
import numpy as np

start = time.time()
reps = 10000
players = [CiroPlayer(0), CiroPlayer(1), CiroPlayer(2), CiroPlayer(3)]
stat_winners = np.zeros((reps, len(players)))

for t in range(reps):
    game = PlayersGame(players, r=5, e=-1, s=0)
    game.play(verbose=0)
    stats = game.return_stats()
    stat_winners[t, :] = stats["winners"]

for i in range(len(players)):
    wins = np.sum(stat_winners[:, i] == 1)
    print(f"{i+1}° Player: Total wins: {wins}")

end = time.time()
print(f"Tempo di esecuzione: {end - start:.6f} secondi")