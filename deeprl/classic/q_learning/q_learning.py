import io
import numpy as np
import torch as th
import pathlib

from gymnasium import spaces

from typing import TypeVar, Union, Type, Dict, Any, Optional, Tuple, ClassVar, Iterable
from deeprl.common.classic_agent import ClassicAgent
from deeprl.common.type_aliases import GymEnv, Schedule, MaybeCallback
from deeprl.classic.q_learning.policies import QTable
from deeprl.common.save_util import save_to_zip_file, load_from_zip_file
from deeprl.common.policies import BasePolicy, BaseTabularPolicy
from deeprl.common.utils import get_linear_fn

SelfQLearning = TypeVar('SelfQLearning', bound='QLearning')

class QLearning(ClassicAgent):
    """
    Q-Learning algorithm.
    
    :param policy: The policy model to use (Tabular or Approximators).
    :param env: The environment to learn from.
        (if registered in Gym, can be str. Can be None for loading trained models)
    :param learning_rate: If you are using aproximators, the learning rate for the optimizer
    :param alpha: The learning rate for the Q-Table.
    :param gamma: The discount factor.
    :param exploration_fraction: fraction of entire training period over which the exploration rate is reduced
    :param exploration_initial_eps: initial value of random action probability
    :param exploration_final_eps: final value of random action probability
    :param stats_window_size: Window size for the rollout logging, specifying the number of episodes to average
        the reported success rate, mean episode length, and mean reward over
    :param tensorboard_log: the log location for tensorboard (if None, no logging)
    :param verbose: Verbosity level: 0 for no output, 1 for info messages (such as device or wrappers used), 2 for
        debug messages
    :param device: Device on which the code should run.
        By default, it will try to use a Cuda compatible device and fallback to cpu
        if it is not possible.
    :param seed: Seed for the pseudo random generators
    """
    policy_aliases: ClassVar[Dict[str, Union[Type[BasePolicy]]]] = {
        "QTable": QTable,
        #"CnnPolicy": CnnPolicy,
        #"MultiInputPolicy": MultiInputPolicy,
    }
    # Linear schedule will be defined in `_setup_model()`
    exploration_schedule: Schedule
    policy: QTable
    
    def __init__(
        self,
        policy: Union[str, Type[BasePolicy], Type[BaseTabularPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule],
        gamma: float = 0.99,
        exploration_fraction: float = 0.1,
        exploration_initial_eps: float = 1.0,
        exploration_final_eps: float = 0.05,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        verbose: int = 0,
        device: Union[th.device, str] = None,
        seed: Optional[int] = None,
        _init_setup_model: bool = True,
    ) -> None:
        super().__init__(
            policy=policy,
            env=env,
            learning_rate=learning_rate,
            gamma=gamma,
            policy_kwargs=policy_kwargs,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            device=device,
            seed=seed,
            supported_action_spaces=(spaces.Discrete,)
        )
        
        self.exploration_initial_eps = exploration_initial_eps
        self.exploration_final_eps = exploration_final_eps
        self.exploration_fraction = exploration_fraction
        self.exploration_rate = 0.0
        self.gamma = gamma
        
        if _init_setup_model:
            self._setup_model()
        
    def _setup_model(self) -> None:
        super()._setup_model()
        self._create_aliases()
        self.exploration_schedule = get_linear_fn(
            self.exploration_initial_eps,
            self.exploration_final_eps,
            self.exploration_fraction
        )
    
    def _create_aliases(self) -> None:
        self.q_table = self.policy.table
    
    def _on_step(self) -> None:
        self.exploration_rate = self.exploration_schedule(self._current_progress_remaining)
        self.logger.record("exploration_rate", self.exploration_rate)
        
    def train(self, state, action, reward, next_state, done):
        """
        Actualiza la tabla Q usando la fórmula de Q-learning.
        """
        state = state.item() if isinstance(state, np.ndarray) else state



        q_value = self.q_table[state][action]
        max_next_q = np.max(self.q_table[next_state]) if not done else 0
        new_q = q_value + self.learning_rate * (reward + self.gamma * max_next_q - q_value)
        self.q_table[state][action] = new_q

    
    def predict(
        self,
        observation: Union[np.ndarray, Dict[str, np.ndarray]],
        state: Optional[Tuple[np.ndarray, ...]] = None,
        episode_start: Optional[bool] = None,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        """
        Overrides the base_class predict function to include epsilon-greedy exploration.

        :param observation: the input observation
        :param state: The last states (can be None, used in recurrent policies)
        :param episode_start: The last masks (can be None, used in recurrent policies)
        :param deterministic: Whether or not to return deterministic actions.
        :return: the model's action and the next state
            (used in recurrent policies)
        """
        if not deterministic and np.random.rand() < self.exploration_rate:
            if self.policy.is_vectorized_observation(observation):
                if isinstance(observation, dict):
                    n_batch = observation[next(iter(observation.keys()))].shape[0]
                else:
                    n_batch = observation.shape[0]
                action = np.array([self.action_space.sample() for _ in range(n_batch)])
            else:
                action = np.array(self.action_space.sample())
        else:
            action = self.policy.predict(observation,deterministic)
        return action
    
    def learn(
        self: SelfQLearning,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 4,
        tb_log_name: str = "QLearning",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfQLearning:
        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )
        
    def save(
        self,
        path: Union[str, pathlib.Path],
        exclude: Optional[Iterable[str]] = None,
        include: Optional[Iterable[str]] = None,
    ) -> None:
        """
        Save the attributes of the QLearning agent and the Q-table in a zip file.

        :param path: Path to the file where the QLearning agent should be saved.
        :param exclude: Parameters to exclude from saving (if any).
        :param include: Parameters to explicitly include even if excluded by default.
        """
        # Copy the agent's attributes
        data = self.__dict__.copy()

        # Handle exclusions
        if exclude is None:
            exclude = []
        exclude = set(exclude).union(self._excluded_save_params())

        if include is not None:
            exclude = exclude.difference(include)

        # Remove excluded parameters
        for param_name in exclude:
            data.pop(param_name, None)

        # Ensure the Q-table is included
        data['q_table'] = self.q_table

        # Save using the helper function
        save_to_zip_file(path, data=data, params=None, pytorch_variables=None)
    
    
    @classmethod
    def load(
        cls: Type[SelfQLearning],
        path: Union[str, pathlib.Path, io.BufferedIOBase],
        env: Optional[GymEnv] = None,
        custom_objects: Optional[Dict[str, Any]] = None,
        print_system_info: bool = False,
        force_reset: bool = True,
        **kwargs,
    ) -> "QLearning":
        """
        Load a QLearning agent and its Q-table from a saved file.

        :param path: Path to the file where the QLearning agent is saved.
        :param env: Optionally, provide a new environment for the agent.
        :param custom_objects: Optional dictionary to replace attributes during loading.
        :param print_system_info: Whether to print system information during loading.
        :param force_reset: Whether to reset the environment upon loading.
        :param kwargs: Additional arguments to override saved attributes.
        :return: Loaded QLearning agent.
        """
        from deeprl.common.save_util import load_from_zip_file

        # Load data and parameters from the zip file
        data, _, _ = load_from_zip_file(
            path,
            custom_objects=custom_objects,
            print_system_info=print_system_info,
        )

        assert data is not None, "No data found in the saved file"
        assert "q_table" in data, "Q-table missing from the saved file"

        # If the environment is provided, ensure compatibility
        if env is not None:
            from deeprl.common.utils import check_for_correct_spaces
            check_for_correct_spaces(env, data["observation_space"], data["action_space"])
            data["env"] = env
            if force_reset:
                env.reset()

        # Initialize the model without setting up the environment
        model = cls(
            policy=data.get("policy_class"),
            env=env,
            learning_rate=data.get("learning_rate"),
            gamma=data.get("gamma"),
            exploration_fraction=data.get("exploration_fraction"),
            exploration_initial_eps=data.get("exploration_initial_eps"),
            exploration_final_eps=data.get("exploration_final_eps"),
            policy_kwargs=data.get("policy_kwargs"),
            stats_window_size=data.get("stats_window_size"),
            tensorboard_log=data.get("tensorboard_log"),
            verbose=data.get("verbose"),
            seed=data.get("seed"),
            _init_setup_model=False,  # Prevent model setup here
        )

        # Restore attributes from the loaded data
        model.__dict__.update(data)
        model.__dict__.update(kwargs)

        # Setup the policy and other components
        model._setup_model()

        # Restore the Q-table directly into the policy (for tabular policies)
        if "q_table" in data and hasattr(model.policy, "table"):
            model.policy.table = data["q_table"]
            model.q_table = data["q_table"]
            
        return model