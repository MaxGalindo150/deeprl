import warnings

import cloudpickle
import gymnasium
from pettingzoo.utils.env import ParallelEnv
from .pettingzoo_vector_wrapper import PettingZooVecEnvWrapper
from .constructors import MakeCPUAsyncConstructor
from .markov_vector_wrapper import MarkovVectorEnv

def vec_env_args(env, num_envs):
    def env_fn():
        env_copy = cloudpickle.loads(cloudpickle.dumps(env))
        return env_copy

    return [env_fn] * num_envs, env.observation_space, env.action_space


def warn_not_gym_env(env, fn_name):
    if not isinstance(env, gymnasium.Env):
        warnings.warn(
            f"{fn_name} took in an environment which does not inherit from gymnasium.Env. Note that gym_vec_env only takes in gymnasium-style environments, not pettingzoo environments."
        )


def gym_vec_env_v0(env, num_envs, multiprocessing=False):
    warn_not_gym_env(env, "gym_vec_env")
    args = vec_env_args(env, num_envs)
    constructor = (
        gymnasium.vector.AsyncVectorEnv
        if multiprocessing
        else gymnasium.vector.SyncVectorEnv
    )
    return constructor(*args)


def deeprl_vec_env_v0(env, num_envs, multiprocessing=False):
    import deeprl

    warn_not_gym_env(env, "deeprl_vec_env")
    args = vec_env_args(env, num_envs)[:1]
    constructor = (
        deeprl.common.vec_env.SubprocVecEnv
        if multiprocessing
        else deeprl.common.vec_env.DummyVecEnv
    )
    return constructor(*args)


def concat_vec_envs_v1(vec_env, num_vec_envs, num_cpus=0):
    num_cpus = min(num_cpus, num_vec_envs)
    vec_env = MakeCPUAsyncConstructor(num_cpus)(*vec_env_args(vec_env, num_vec_envs))
        
    return PettingZooVecEnvWrapper(vec_env)
    

def pettingzoo_env_to_vec_env_v1(parallel_env):
    assert isinstance(
        parallel_env, ParallelEnv
    ), "pettingzoo_env_to_vec_env takes in a pettingzoo ParallelEnv. Can create a parallel_env with pistonball.parallel_env() or convert it from an AEC env with `from pettingzoo.utils.conversions import aec_to_parallel; aec_to_parallel(env)``"
    assert hasattr(
        parallel_env, "possible_agents"
    ), "environment passed to pettingzoo_env_to_vec_env must have possible_agents attribute."
    return MarkovVectorEnv(parallel_env)
