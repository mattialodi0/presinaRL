import gym
from presina_env import PresinaEnv

def play_presina():
    # Create an instance of the Presina environment
    env = PresinaEnv()

    # Reset the environment to start a new game
    obs, info = env.reset()

    done = False
    while not done:
        # Render the current state of the game
        env.render()

        # Sample a random action (decide whether to take the card)
        action = env.action_space.sample()

        # Step through the environment with the chosen action
        obs, reward, done, info = env.step(action)

        # Print the result of the action
        print(f"Action taken: {action}, Reward: {reward}, Done: {done}")

    # Close the environment
    env.close()

if __name__ == "__main__":
    play_presina()