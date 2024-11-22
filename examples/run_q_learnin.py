import gymnasium as gym

from deeprl import QLearning

env = gym.make("FrozenLake-v1", is_slippery=False)

agent = QLearning(
    policy="TabularPolicy",
    env=env,
    gamma=0.99,
    exploration_fraction=0.2,
    exploration_initial_eps=1.0,
    exploration_final_eps=0.05,
    verbose=2,
)

agent.learn(total_timesteps=100)
