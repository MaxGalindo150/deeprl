import sys
import time
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
    :param stats_window_size: Window size for the rollout logging, specifying the number of episodes 
        to average the reported success rate, mean episode length, and mean reward over
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
        assert self.env is not None, "You must set the environment before calling _setup_learn()"
        
        return super()._setup_learn(
            total_timesteps,
            callback,
            reset_num_timesteps,
            tb_log_name,
            progress_bar,
        )
    
    def learn(
        self: SelfClassicAgent,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 4,
        tb_log_name: str = "run",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfClassicAgent:
        
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
            rollout = self.collect_rollouts(
                self.env,
                callback,
                log_interval,
            )
            
            if not rollout.continue_training:
                break
            
            self.train(rollout)
        
        callback.on_training_end()
        
        return self 
            
    def train(self, rollout: RolloutReturn) -> None:
        """
        Train the agent from a buffer of rollouts.
        
        :param rollout: The collected rollout.
        """
        raise NotImplementedError()
        
    def _sample_action(self, obs: np.ndarray) -> int:
        """
        Sample an action using an epsilon-greedy policy with function approximation.

        :param obs: Current observation (state).
        :return: Action to take in the environment.
        """
        if np.random.rand() < self.epsilon:  
            return self.action_space.sample()
        else:
            q_values = self.policy.predict(obs)
            return np.argmax(q_values)  

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
        
    def _on_step(self) -> None:
        """
        Method called after each step in the environment.
        It is meant to trigger DQN target network update
        but can be used for other purposes
        """
        pass
    
    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        log_interval: Optional[int] = None,
    ) -> RolloutReturn:
        """
        Collect experiences for classical RL algorithms.

        :param env: The training environment.
        :param callback: Callback that will be called at each step and at the beginning and end of the rollout.
        :param log_interval: Log data every ``log_interval`` episodes.
        :return: RolloutReturn containing the number of collected steps and episodes.
        """
        # self.policy.set_training_mode(False) ----> No se necesita en este caso
        num_collected_steps, num_collected_episodes = 0, 0

        assert isinstance(env, VecEnv), "You must pass a VecEnv"

        callback.on_rollout_start()
        continue_training = True

        while continue_training:
            # Get the agent's action from the observation
            action = self._sample_action(self._last_obs)

            # Take a step in the environment
            new_obs, reward, done, info = env.step(action)

            # Update the total number of steps
            self.num_timesteps += 1
            num_collected_steps += 1

            # Call the callback with the training update
            callback.update_locals(locals())
            if not callback.on_step():
                continue_training = False
                break

            # # Actualizar la tabla Q o la función de aproximación
            # self._store_transition(self._last_obs, action, reward, new_obs, done)

            # Si se termina el episodio
            if done:
                num_collected_episodes += 1
                self._episode_num += 1

                # Reiniciar el entorno
                self._last_obs = env.reset()

                # Loggear estadísticas si es necesario
                if log_interval is not None and self._episode_num % log_interval == 0:
                    self._dump_logs()

            # Si el episodio no termina, actualizar observación actual
            else:
                self._last_obs = new_obs

        callback.on_rollout_end()

        return RolloutReturn(num_collected_steps, num_collected_episodes, continue_training)

    
    
        
        
        
        
        