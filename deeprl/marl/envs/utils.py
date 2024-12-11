import supersuit as ss
from gymnasium.spaces import Box
from pettingzoo.utils.conversions import parallel_wrapper_fn
from deeprl.marl.common.vec_env.vector_constructors import concat_vec_envs_v1


def make_marl_env(env_fn, num_vec_envs=8, num_cpus=1, visual_preprocessing=True, **env_kwargs):
    """
    Generalized wrapper to create a multi-agent environment compatible with vectorized training.
    
    
    :param env_fn: The PettingZoo environment constructor.
    :param num_vec_envs: Number of parallel instances of the vectorized environment.
    :param num_cpus: Number of CPUs to use for parallelization.
    :param visual_preprocessing: Whether to preprocess visual observations.
    :param env_kwargs: Additional arguments for the environment constructor.
    """
    # Initialize the parallel environment
    env = env_fn.parallel_env(**env_kwargs)
    
    # Add black death wrapper to ensure consistent agent count
    env = ss.black_death_v3(env)

    # Detect visual observations
    is_visual = isinstance(env.observation_space(env.possible_agents[0]), Box) and \
                len(env.observation_space(env.possible_agents[0]).shape) == 3
    
    # Apply visual preprocessing if enabled
    if is_visual and visual_preprocessing:
        env = ss.color_reduction_v0(env, mode="B")
        env = ss.resize_v1(env, x_size=84, y_size=84)
        env = ss.frame_stack_v1(env, 3)
    
    # Convert to vectorized environment
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = concat_vec_envs_v1(env, num_vec_envs=num_vec_envs, num_cpus=num_cpus)

    return env
