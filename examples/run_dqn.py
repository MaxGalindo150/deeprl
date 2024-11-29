import gymnasium as gym

from deeprl import DQN

env = gym.make("CartPole-v1")

model = DQN(
    policy="MlpPolicy", 
    env=env, 
    tensorboard_log="./dqn_cartpole_tensorboard/", 
    verbose=1)

model.learn(total_timesteps=1_000_000, log_interval=4)
model.save("dqn_cartpole")

del model # remove to demonstrate saving and loading

model = DQN.load("dqn_cartpole")

obs, info = env.reset()
while True:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()