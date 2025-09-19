from GameVariants import PlayersGame
from Deck import Deck, Card
from players.CiroPlayer import CiroPlayer
from PlayerNode import goto_player
import numpy as np
import math

reps = 100000

for r in range(5):
    print(f"--- Round with {5-r} cards: ---")

    print("With Ciro Playstyle:")
    players = [CiroPlayer(0), CiroPlayer(1), CiroPlayer(2), CiroPlayer(3)]
    stat_winners = np.zeros((reps, len(players)))

    for t in range(reps):
        game = PlayersGame(players, r=1, e=-1, s=r)
        game.play(verbose=0)
        stats = game.return_stats()
        stat_winners[t, :] = stats["winners"]

    for i in range(len(players)):
        wins = np.sum(stat_winners[:, i] == 1)
        print(f"{i+1}° Player: Total wins: {wins}")