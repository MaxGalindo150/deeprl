import os

import gymnasium as gym
import ale_py

gym.register_envs(ale_py)

from deeprl import DQN
from deeprl.common.monitor import Monitor
from deeprl.common.vec_env import DummyVecEnv, VecFrameStack


log_dir = "examples/dqn/logs"
os.makedirs(log_dir, exist_ok=True)

seeds = [0, 42, 100]

env_id = "ALE/SpaceInvaders-v5"

for seed in seeds:
    def make_env():
        env = gym.make(env_id)
        env.reset(seed=seed)  
        log_file = os.path.join(log_dir, f"deeprl_{seed}.csv")
        env = Monitor(env, log_file)
        return env

    vec_env = DummyVecEnv([make_env])
    vec_env = VecFrameStack(vec_env, n_stack=4)

    model = DQN("CnnPolicy", vec_env, buffer_size=33_000, seed=seed)
    print(f"Training with seed {seed}...")
    model.learn(total_timesteps=int(1e7), progress_bar=True)

    model_path = os.path.join(log_dir, f"dqn_spaceinvaders_seed_{seed}.zip")
    model.save(model_path)
    print(f"Model saved at {model_path}")
    del model
    vec_env.close()

print("Training complete for all seeds.")
