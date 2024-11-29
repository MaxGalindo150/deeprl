import io
import pathlib
import sys
import time
from typing import TypeVar, Union, Type, Dict, Any, Optional, Tuple

import torch as th
from torch import nn
import numpy as np
from gymnasium import spaces
from deeprl.common.save_util import load_from_zip_file, recursive_getattr, recursive_setattr, save_to_zip_file
from deeprl.common.utils import safe_mean, should_collect_more_steps
from deeprl.common.callbacks import BaseCallback
from deeprl.common.base_agent import BaseAgent
from deeprl.common.type_aliases import GymEnv, Schedule, MaybeCallback
from deeprl.common.policies import BasePolicy, BaseTabularPolicy
from deeprl.common.utils import (
    check_for_correct_spaces
)
from deeprl.common.vec_env.patch_gym import _convert_space, _patch_env


SelfClassicAgent = TypeVar('SelfClassicAgent', bound='ClassicAgent')

class ClassicAgent(BaseAgent):
    """
    The base classs for Classic Agents (e.g. Q-Learning, SARSA, etc.)
    
    :param policy: The policy model to use.
    :param env: The environment to lear from.
        (if registered in Gym, can be str. Can be None for loading trained models)
    :param learning_rate: Learning rate for the optimizer
    :param gamma: The discount factor
    :param policy_kwargs: Additional arguments to be passed to the policy on creation
    :param stats_window_size: Window size for the rollout logging, specifying the number of episodes to average
        the reported success rate, mean episode length, and mean reward over
    :param tensorboard_log: the log location for tensorboard (if None, no logging)
    :param verbose: Verbosity level: 0 for no output, 1 for info messages (such as device or wrappers used), 2 for
        debug messages
    :param device: Device on which the code should run.
        By default, it will try to use a Cuda compatible device and fallback to cpu
        if it is not possible.
    :param seed: Seed for the pseudo random generators
    :param supported_action_spaces: The action spaces supported by the algorithm.
    """
    
    def __init__(
        self,
        policy: Union[str, Type[BasePolicy], Type[BaseTabularPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule],
        gamma: float = 0.99,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        verbose: int = 0,
        device: Union[th.device, str] = None,
        seed: Optional[int] = None,
        supported_action_spaces: Optional[Tuple[Type[spaces.Space], ...]] = None,
    ):
        super().__init__(
            policy=policy,
            env=env,
            learning_rate=learning_rate,
            policy_kwargs=policy_kwargs,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            device="auto",
            support_multi_env=False,
            monitor_wrapper=True,
            seed=seed,
            use_sde=False,
            sde_sample_freq=-1,
            supported_action_spaces=supported_action_spaces
        )
        
    def _setup_model(self) -> None:
        self._setup_lr_schedule()
        self.set_random_seed(self.seed)
        
        self.policy = self.policy_class(
            self.observation_space,
            self.action_space,
            self.lr_schedule,
            **self.policy_kwargs
        )
        
        if isinstance(self.policy, nn.Module):
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

        assert self.env is not None, "You must set the `env` attribute before calling _setup_learn()"
        
        return super()._setup_learn(
            total_timesteps,
            callback,
            reset_num_timesteps,
            tb_log_name,
            progress_bar
        )
    
    def learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 4,
        tb_log_name: str = "run",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> "SelfClassicAgent":
        # Configuración inicial
        total_timesteps, callback = self._setup_learn(
            total_timesteps,
            callback,
            reset_num_timesteps,
            tb_log_name,
            progress_bar,
        )

        callback.on_training_start(locals(), globals())

        assert self.env is not None, "You must set the environment before calling learn()"

        while self.num_timesteps < total_timesteps:
            
            current_state = self.env.reset()
            done = False

            while not done:
                action = self.predict(current_state, self.policy.table, self.num_timesteps)

                next_state, reward, done, info = self.env.step(action)

                self.train(current_state, action, reward, next_state, done)

                current_state = next_state
                self.num_timesteps += 1
                self._update_info_buffer(info, done)
                # self._update_current_progress_remaining(self.num_timesteps, self._total_timesteps)

                if callback:
                    callback.on_step()

                if log_interval and self.num_timesteps % log_interval == 0:
                    self._dump_logs()

                if self.num_timesteps >= total_timesteps:
                    done = True
                if done:
                    self._episode_num += 1
                    
  
                    
            

            # if callback:
            #     callback.on_episode_end()

        callback.on_training_end()

        return self

    def train(self, state, action, reward, next_state, done):
        """
        This method must be implemented by subclasses. It should contain the logic to train the agent.
        """
        raise NotImplementedError("The train method must be implemented by subclasses.")


    def _on_step(self) -> None:
        """
        Method called after each step in the environment.
        """
        pass
        
    @classmethod
    def load(
        cls: Type[SelfClassicAgent],
        path: Union[str, pathlib.Path, io.BufferedIOBase],
        env: Optional[GymEnv] = None,
        device: Union[th.device, str] = "auto",
        custom_objects: Optional[Dict[str, Any]] = None,
        print_system_info: bool = False,
        force_reset: bool = True,
        **kwargs,
    ) -> "SelfClassicAgent":
        data, params, pytorch_variables = load_from_zip_file(
            path,
            device=device,
            custom_objects=custom_objects,
            print_system_info=print_system_info,
        ) 

        assert data is not None, "No data found in the saved file"
        assert params is not None, "No params found in the saved file"

        if "policy_kwargs" in data:
            del data["policy_kwargs"]

        if "observation_space" not in data or "action_space" not in data:
            raise KeyError("The observation_space and action_space were not given, can't verify new environments")

        # Gym -> Gymnasium space conversion
        for key in {"observation_space", "action_space"}:
            data[key] = _convert_space(data[key])
            
        if env is not None:
            # Ensure the environment is wrapped and spaces are correct
            env = cls._wrap_env(env, data["verbose"])
            check_for_correct_spaces(env, data["observation_space"], data["action_space"])
            if force_reset:
                data["_last_obs"] = None
            data["n_envs"] = env.num_envs
        else:
            if "env" in data:
                env = data["env"]

        model = cls(
            policy=data["policy_class"],
            env=env,
            device=device,
            _init_setup_model=False,  # Classic algorithms do not rely on NN setup
        )

        # Load parameters specific to tabular models
        model.__dict__.update(data)
        model.__dict__.update(kwargs)
        model._setup_model()
        # Custom setup for tabular models
        if isinstance(model.policy, BaseTabularPolicy):  # Example for Q-tables
            model.policy.table = params["table"]

        return model

    def _dump_logs(self) -> None:
        """
        Write log.
        """
        assert self.ep_info_buffer is not None
        assert self.ep_success_buffer is not None

        time_elapsed = max((time.time_ns() - self.start_time) / 1e9, sys.float_info.epsilon)
        fps = int((self.num_timesteps - self._num_timesteps_at_start) / time_elapsed)
        self.logger.record("time/episodes", self._episode_num, exclude="tensorboard")
        if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
            self.logger.record("rollout/ep_rew_mean", safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer]))
            self.logger.record("rollout/ep_len_mean", safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer]))
        self.logger.record("time/fps", fps)
        self.logger.record("time/time_elapsed", int(time_elapsed), exclude="tensorboard")
        self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
        if self.use_sde:
            self.logger.record("train/std", (self.actor.get_std()).mean().item())

        if len(self.ep_success_buffer) > 0:
            self.logger.record("rollout/success_rate", safe_mean(self.ep_success_buffer))
        # Pass the number of timesteps for tensorboard
        self.logger.dump(step=self.num_timesteps)
        