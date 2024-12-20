import gymnasium as gym
from deeprl import QLearning

env = gym.make("Taxi-v3")

model = QLearning(
    policy="QTable",
    env=env,
    learning_rate=0.1,
    gamma=0.99,
    verbose=2
)

model.learn(total_timesteps=1_000_000, log_interval=4)

model.save("frozenlake_q_table")

del model # remove to demonstrate saving and loading

model = QLearning.load("frozenlake_q_table")


env = gym.make("Taxi-v3", render_mode="human")


obs, info = env.reset()
while True:
    action = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()
