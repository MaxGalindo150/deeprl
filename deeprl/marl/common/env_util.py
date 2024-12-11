import pettingzoo

from typing import Any, Callable, Optional, Type, Union

from deeprl.common.vec_env import DummyVecEnv, SubprocVecEnv

def make_marl_vec_env(
    env_fn: Union[pettingzoo.AECEnv, pettingzoo.ParallelEnv],
    n_envs: int = 1,
    seed: Optional[int] = None,
    start_index: int = 0,
    monitor_dir: Optional[str] = None,
    wrapper_class: Optional[Callable[[gym.Env], gym.Env]] = None,
    env_kwargs: Optional[dict[str, Any]] = None,
    vec_env_cls: Optional[Type[Union[DummyVecEnv, SubprocVecEnv]]] = None,
    vec_env_kwargs: Optional[dict[str, Any]] = None,
    monitor_kwargs: Optional[dict[str, Any]] = None,
    wrapper_kwargs: Optional[dict[str, Any]] = None,
    is_atari: bool = False,
)-> VecEnv: