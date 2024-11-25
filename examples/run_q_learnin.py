import gymnasium as gym
from deeprl import QLearning

env = gym.make("FrozenLake-v1", is_slippery=True)

model = QLearning(
    policy="QTable",
    env=env,
    learning_rate=0.1,
    gamma=0.99,
    verbose=2,
    tensorboard_log="./q_learning_frozenlake_tensorboard/",
)

model.learn(total_timesteps=1_000_000, log_interval=4)

model.save("frozenlake_q_table")

# del model # remove to demonstrate saving and loading

# model = QLearning.load("frozenlake_q_table")

env = gym.make("FrozenLake-v1", is_slippery=True, render_mode="human")


obs, info = env.reset()
while True:
    action = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()
