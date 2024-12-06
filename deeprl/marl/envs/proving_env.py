from deeprl.common.vec_env import SubprocVecEnv
from pettingzoo.mpe import simple_spread_v3

def make_env():
    def _init():
        return simple_spread_v3.parallel_env(render_mode="human")
    return _init

if __name__ == "__main__":
    vec_env = SubprocVecEnv([make_env() for _ in range(4)])  # 4 entornos paralelos
    obs = vec_env.reset()
    actions = {agent: vec_env.action_space(agent).sample() for agent in vec_env.envs[0].agents}
    obs, rewards, dones, infos = vec_env.step(actions)
    vec_env.close()
