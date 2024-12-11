from __future__ import annotations

import glob
import os
import time

import supersuit as ss
from deeprl import PPO
from deeprl.deep.ppo import CnnPolicy, MlpPolicy
from deeprl.marl.common.vec_env.vector_constructors import concat_vec_envs_v1

from pettingzoo.mpe import simple_spread_v3


def train(env_fn, steps: int = 10_000, seed: int | None = 0, **env_kwargs):
    # Train a single model to play as each agent in an AEC environment
    env = env_fn.parallel_env(**env_kwargs)

    # Add black death wrapper so the number of agents stays constant
    env = ss.black_death_v3(env)

    from gymnasium.spaces import Box
    visual_observation = isinstance(env.observation_space(env.possible_agents[0]), Box) and len(env.observation_space(env.possible_agents[0]).shape) == 3

    if visual_observation:
        env = ss.color_reduction_v0(env, mode="B")
        env = ss.resize_v1(env, x_size=84, y_size=84)
        env = ss.frame_stack_v1(env, 3)

    env.reset(seed=seed)

    print(f"Starting training on {str(env.metadata['name'])}.")

    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = concat_vec_envs_v1(env, 8, num_cpus=1)

    # Use a CNN policy if the observation space is visual
    model = PPO(
        CnnPolicy if visual_observation else MlpPolicy,
        env,
        verbose=3,
        batch_size=256,
        device="cpu",
    )

    model.learn(total_timesteps=steps)

    model.save(f"{env.unwrapped.metadata.get('name')}_{time.strftime('%Y%m%d-%H%M%S')}")

    print("Model has been saved.")

    print(f"Finished training on {str(env.unwrapped.metadata['name'])}.")

    env.close()


def eval(env_fn, num_games: int = 100, render_mode: str | None = None, **env_kwargs):
    # Evaluate a trained agent vs a random agent
    env = env_fn.env(render_mode=render_mode, **env_kwargs)

    from gymnasium.spaces import Box
    visual_observation = isinstance(env.observation_space(env.possible_agents[0]), Box) and len(env.observation_space(env.possible_agents[0]).shape) == 3

    if visual_observation:
        env = ss.color_reduction_v0(env, mode="B")
        env = ss.resize_v1(env, x_size=84, y_size=84)
        env = ss.frame_stack_v1(env, 3)

    print(
        f"\nStarting evaluation on {str(env.metadata['name'])} (num_games={num_games}, render_mode={render_mode})"
    )

    try:
        latest_policy = max(
            glob.glob(f"{env.metadata['name']}*.zip"), key=os.path.getctime
        )
    except ValueError:
        print("Policy not found.")
        exit(0)

    model = PPO.load(latest_policy, device="cpu")

    rewards = {agent: 0 for agent in env.possible_agents}

    # Note: we evaluate here using an AEC environments, to allow for easy A/B testing against random policies
    for i in range(num_games):
        env.reset(seed=i)
        env.action_space(env.possible_agents[0]).seed(i)

        for agent in env.agent_iter():
            obs, reward, termination, truncation, info = env.last()

            for a in env.agents:
                rewards[a] += env.rewards[a]

            if termination or truncation:
                break
            else:
                if agent == env.possible_agents[0]:
                    act = env.action_space(agent).sample()
                else:
                    act = model.predict(obs, deterministic=True)[0]
            env.step(act)
    env.close()

    avg_reward = sum(rewards.values()) / len(rewards.values())
    avg_reward_per_agent = {
        agent: rewards[agent] / num_games for agent in env.possible_agents
    }
    print(f"Avg reward: {avg_reward}")
    print("Avg reward per agent, per game: ", avg_reward_per_agent)
    print("Full rewards: ", rewards)
    return avg_reward


if __name__ == "__main__":
    env_fn = simple_spread_v3

    # Set vector_state to false in order to use visual observations (significantly longer training time)
    env_kwargs = dict(N=3, local_ratio=0.5, max_cycles=25)

    # Train a model
    train(env_fn, steps=81_920, seed=0, **env_kwargs)

    # Evaluate 10 games
    eval(env_fn, num_games=10, render_mode=None, **env_kwargs)

    # Watch 2 games
    eval(env_fn, num_games=2, render_mode="human", **env_kwargs)
