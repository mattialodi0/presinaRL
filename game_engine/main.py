from Game import Game, PlayableGame

if __name__ == "__main__":
    game = Game(r=1, s=4)
    # game = PlayableGame(r=1)
    game.play(verbose=True)
    
    # stats = game.return_stats()
    # print(stats)
