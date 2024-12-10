import time
import glob
import os
from deeprl.marl.envs.utils import make_marl_env
from deeprl import PPO
from gymnasium.spaces import Box
from deeprl.deep.ppo import CnnPolicy, MlpPolicy

from pettingzoo.mpe import simple_spread_v3

env_fn = simple_spread_v3
env_kwargs = dict(N=3, local_ratio=0.5, max_cycles=25)

env = make_marl_env(env_fn, num_vec_envs=8, num_cpus=1, **env_kwargs)


print(f"Starting training on {str(env.unwrapped.metadata['name'])}.")

# Detecta si el entorno utiliza observaciones visuales
is_visual = isinstance(env.observation_space, Box) and len(env.observation_space.shape) == 3

# Inicializa el modelo PPO
model = PPO(
    CnnPolicy if is_visual else MlpPolicy,
    env,
    verbose=3,
    batch_size=256,
    device="cpu",
)

model.learn(total_timesteps=1_000_000)

# Guarda el modelo entrenado
model.save(f"{env.unwrapped.metadata.get('name')}_{time.strftime('%Y%m%d-%H%M%S')}")
print("Model has been saved.")
env.close()





# def eval(env_fn, num_games: int = 100, render_mode: str | None = None, **env_kwargs):
#     # Evaluate a trained agent vs a random agent
#     env = env_fn.env(render_mode=render_mode, **env_kwargs)

#     from gymnasium.spaces import Box
#     visual_observation = isinstance(env.observation_space(env.possible_agents[0]), Box) and len(env.observation_space(env.possible_agents[0]).shape) == 3

#     if visual_observation:
#         env = ss.color_reduction_v0(env, mode="B")
#         env = ss.resize_v1(env, x_size=84, y_size=84)
#         env = ss.frame_stack_v1(env, 3)

#     print(
#         f"\nStarting evaluation on {str(env.metadata['name'])} (num_games={num_games}, render_mode={render_mode})"
#     )

#     try:
#         latest_policy = max(
#             glob.glob(f"{env.metadata['name']}*.zip"), key=os.path.getctime
#         )
#     except ValueError:
#         print("Policy not found.")
#         exit(0)

#     model = PPO.load(latest_policy, device="cpu")

#     rewards = {agent: 0 for agent in env.possible_agents}

#     # Note: we evaluate here using an AEC environments, to allow for easy A/B testing against random policies
#     for i in range(num_games):
#         env.reset(seed=i)
#         env.action_space(env.possible_agents[0]).seed(i)

#         for agent in env.agent_iter():
#             obs, reward, termination, truncation, info = env.last()

#             for a in env.agents:
#                 rewards[a] += env.rewards[a]

#             if termination or truncation:
#                 break
#             else:
#                 if agent == env.possible_agents[0]:
#                     act = env.action_space(agent).sample()
#                 else:
#                     act = model.predict(obs, deterministic=True)[0]
#             env.step(act)
#     env.close()

#     avg_reward = sum(rewards.values()) / len(rewards.values())
#     avg_reward_per_agent = {
#         agent: rewards[agent] / num_games for agent in env.possible_agents
#     }
#     print(f"Avg reward: {avg_reward}")
#     print("Avg reward per agent, per game: ", avg_reward_per_agent)
#     print("Full rewards: ", rewards)
#     return avg_reward


# env_fn = simple_spread_v3
# env_kwargs = dict(N=3, local_ratio=0.5, max_cycles=25)

# train(env_fn, steps=81_920, seed=0, **env_kwargs)
# eval(env_fn, num_games=10, render_mode="human", **env_kwargs)
