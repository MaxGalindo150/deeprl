import gymnasium as gym

from deeprl import QLearning, QLearningPolicy


env = gym.make("CartPole-v1")
agent = QLearning(policy=QLearningPolicy, env=env)

agent.learn(total_timesteps=10000)

# Evaluar el agente
obs = env.reset()
done = False
while not done:
    action, _ = agent.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()

env.close()