from deeprl.common.env_util import make_atari_env
from deeprl.common.vec_env import VecFrameStack
from deeprl import DQN

# There already exists an environment generator
# that will make and wrap atari environments correctly.
# Here we are also multi-worker training (n_envs=4 => 4 environments)
vec_env = make_atari_env("ALE/SpaceInvaders-v5", n_envs=1, seed=0)
# Frame-stacking with 4 frames
#vec_env = VecFrameStack(vec_env, n_stack=4)

model = DQN.load("examples/dqn/logs/dqn_spaceinvaders_seed_0.zip")

print(model.policy)

obs = vec_env.reset()
while True:
    action, _states = model.predict(obs, deterministic=False)
    obs, rewards, dones, info = vec_env.step(action)
    vec_env.render("human")


