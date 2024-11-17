import io
import pathlib
import sys
import time
import warnings
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union

import numpy as np
import torch as th
from gymnasium import spaces

from deeprl.common.base_agent import BaseAgent
from deeprl.common.callbacks import BaseCallback
from deeprl.common.noise import ActionNoise, VectorizedActionNoise
from deeprl.common.policies import BasePolicy
from deeprl.common.save_util import load_from_pkl, save_to_pkl
from deeprl.common.type_aliases import GymEnv, MaybeCallback, RolloutReturn, Schedule, TrainFreq, TrainFrequencyUnit
from deeprl.common.utils import safe_mean, should_collect_more_steps
from deeprl.common.vec_env import VecEnv

SelfClassicAgent = TypeVar("SelfClassicAgent", bound="ClassicAgent")

class ClassicAgent(BaseAgent):
    """
    The base for all classic agents (ex: Q-Learning, SARSA, Monte Carlo, etc.).
    
    :param policy: The policy model to use (e.g. Q-table)
    :param env: The environment to learn from 
        (if registered in Gym, can be str. Can be None for loading trained models)
    :param gamma: The discount factor
    :param policy_kwargs: Additional arguments to be passed to the policy on creation
    :param stats_window_size: Window size for the rollout logging, specifying the number of episodes to average
        the reported success rate, mean episode length, and mean reward over
    :param tensorboard_log: The log location for tensorboard (if None, no logging)
    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param monitor_wrapper: When creating an environment, whether to wrap it
        or not in a Monitor wrapper.
    :param seed: Seed for the pseudo random generators
    :param supported_action_spaces: The action spaces supported by the algorithm.
    """
    
    def __init__(
        self,
        policy: Union[str, Type[BasePolicy]],
        env: Union[GymEnv, str],
        gamma: float = 0.99,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        verbose: int = 0,
        device: Union[th.device, str] = "auto",
        monitor_wrapper: bool = False,
        seed: Optional[int] = None,
        supported_action_spaces: Optional[List[Type[spaces.Space]]] = None,
    ) -> None:
        super().__init__(
            policy,
            env,
            gamma,
            policy_kwargs=policy_kwargs,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            device=device,
            monitor_wrapper=monitor_wrapper,
            seed=seed,
        )
        self.gamma = gamma
        
    
    def _setup_model(self) -> None:
        self._setup_lr_schedule()
        self.set_random_seed(self.seed)
        
        self.policy = self.policy_class(
            self.observation_space,
            self.action_space,
            self.lr_schedule,
            **self.policy_kwargs,
        )
        self.policy = self.policy.to(self.device)
    
    def _setup_learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        reset_num_timesteps: bool = True,
        tb_log_name: str = "run",
        progress_bar: bool = False,
    ) -> Tuple[int, BaseCallback]:
        """
        cf `BaseAgent`.
        """
        
        
        
        
        