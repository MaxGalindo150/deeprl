import gymnasium as gym

from deeprl import DQN


env = gym.make("LunarLander-v3", render_mode="human")

model = DQN.load("examples/dqn_lunar.zip")

obs, info = env.reset()
while True:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()
    
env.close()
