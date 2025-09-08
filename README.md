# presinaRL
Statistical analysis of a card game aimed to develop a RL agent.

## What is Presina?
Presina is a card game played with italian cards (40 cards divided into 4 seeds of 10 each).
Given n players, at each round r (going from 0 to 4),  (5-r) cards will be distributed to each one.
A total order is imposed on the cards with only one exception (that can be the highest or the lowest, the player choose).
The aim of the game is to predict the number of catches each player makes in every round; a player "catch" when he plays the highest card.
In turn, every player before the round starts states the number of catches he makes, which is called prediction.
The last player to predict cannot state a number that would make the sum of the predictions equal to the 5-r.
Starting from the first player, each one plays a card and when everyone has played it is decided who make the catch. This is then repeated 5-r-1 more times.
Every time a prediction is wrong, a player makes as many error as the score difference, at e errors a player is eliminated.
The last player standing is the winner.
The round with only one card is different, as everyone sees the card of the others but not his own, and there is no constraint
on the predictions.


### 1. Development of a game engine to run simulated matches
### 2. Statistical analysis of the matches
### 3. Developement of am RL agent