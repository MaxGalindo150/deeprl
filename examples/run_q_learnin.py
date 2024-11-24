import gymnasium as gym

from deeprl import QLearning

env = gym.make("FrozenLake-v1", is_slippery=False)

agent = QLearning(
    policy="TabularPolicy",
    env=env,
    learning_rate=0.1,
    gamma=0.99,
    verbose=2,
)

agent.learn(total_timesteps=100)
