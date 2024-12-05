# import os
# import gymnasium as gym

# import numpy as np
# import matplotlib.pyplot as plt

# from deeprl import PPO
# from deeprl.common.vec_env import DummyVecEnv, VecNormalize
# from deeprl.common.monitor import Monitor
# from deeprl.common.results_plotter import load_results, ts2xy

# log_dir = "logs/pusher/"
# os.makedirs(log_dir, exist_ok=True)

# # Crear una función para inicializar el entorno
# def make_env():
#     env = gym.make("Pusher-v5", render_mode="rgb_array")  # Cambia según la versión instalada (v2, v4, o v5)
#     env = Monitor(env, log_dir)  # Registrar recompensas y episodios
#     return env

# # Crear el vector de entornos
# env = DummyVecEnv([make_env])

# # Normalizar observaciones y recompensas
# env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

# # Crear el modelo PPO
# model = PPO(
#     "MlpPolicy",  # Política basada en redes completamente conectadas (MLP)
#     env,
#     n_epochs=10,
#     learning_rate=3e-4,  
#     n_steps=2048,        
#     batch_size=64,       
#     gamma=0.99,          
#     device="cpu"
# )

# # Entrenar el modelo
# model.learn(total_timesteps=3_000_000, progress_bar=True)

# # Guardar el modelo
# model.save("ppo_pusher")

# # Cargar el modelo (si necesitas probarlo más adelante)
# model = PPO.load("ppo_pusher", env=env, device="cpu")

# # Evaluar el agente en el entorno
# obs = env.reset()
# while True:
#     action, _states = model.predict(obs, deterministic=True)
#     obs, reward, done, info = env.step(action)
#     env.render("human")  # Visualizar el entorno
    
# # Cerrar el entorno
# env.close()



# def moving_average(values, window):
#     """
#     Smooth values by doing a moving average
#     :param values: (numpy array)
#     :param window: (int)
#     :return: (numpy array)
#     """
#     weights = np.repeat(1.0, window) / window
#     return np.convolve(values, weights, "valid")


# def plot_results(log_folder, title="Learning Curve"):
#     """
#     plot the results

#     :param log_folder: (str) the save location of the results to plot
#     :param title: (str) the title of the task to plot
#     """
#     x, y = ts2xy(load_results(log_folder), "timesteps")
#     y = moving_average(y, window=50)
#     # Truncate x
#     x = x[len(x) - len(y) :]

#     fig = plt.figure(title)
#     plt.plot(x, y)
#     plt.xlabel("Number of Timesteps")
#     plt.ylabel("Rewards")
#     plt.title(title + " Smoothed")
#     plt.savefig("pusher_learning_curve.png")
    
# # Plot the learning curve
# plot_results(log_dir)


import gymnasium as gym

from deeprl import PPO
from deeprl.common.env_util import make_vec_env
# Parallel environments
vec_env = make_vec_env("CartPole-v1", n_envs=4)
model = PPO("MlpPolicy", vec_env, verbose=1, device="cpu")
model.learn(total_timesteps=25000)
model.save("ppo_cartpole")
del model # remove to demonstrate saving and loading
model = PPO.load("ppo_cartpole")
obs = vec_env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = vec_env.step(action)
    vec_env.render("human")