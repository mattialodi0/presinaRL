from Game import Game, PlayableGame

if __name__ == "__main__":

    # game = Game()
    game = PlayableGame(r=1)
    game.play()
    game.print_stats()
