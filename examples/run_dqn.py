import gymnasium as gym
import imageio
from deeprl import DQN

env = gym.make("CartPole-v1", render_mode="rgb_array")

model = DQN("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=1_000_000)

model.save("cartpole_dqn")

frames = []

vec_env = model.get_env()
obs = vec_env.reset()
for i in range(500):
    frames.append(vec_env.render("rgb_array"))
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec_env.step(action)
    vec_env.render("human")

env.close()

imageio.mimsave("cartpole_dqn.gif", frames, fps=30)
