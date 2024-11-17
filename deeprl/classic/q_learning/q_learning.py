import warnings
from typing import Any, ClassVar, Dict, List, Optional, Tuple, Type, TypeVar, Union

from deeprl.common.off_policy_agent import OffPolicyAgent

SelfQLearning = TypeVar("SelfQLearning", bound="QLearning")

class QLearning(OffPolicyAgent):
    """
    Q-Learning agent.
    
    Reference: Sutton and Barto, Reinforcement Learning: An Introduction, 2nd edition, 2018.
    
    :param policy: The policy model to use (e.g. Q-table)
    :param env: The environment to learn from (if registered in gym, can be str)
    :param gamma: The discount factor
    :param exploration_fraction: Fraction of entire training period over which the exploration rate is reduced
    :param exploration_initial_eps: Initial value of random action probability
    :param exploration_final_eps: Final value of random action probability
    :param stats_window_size: Window size for the rollout logging, specifying the number of episodes to average
        the reported succes rate, mean episode length, and mean reward over
    :param tensorboard_log: The log location for tensorboard (if None, no logging)
    :param policy_kwars: additional arguments to be passed to the policy on creation
    :param verbose: Verbosity level: 0 for no output, 1 for info messages (such as device or wrappers used), 2 for debug messages
    :param seed: Seed for pseudo random generators
    param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
    """
    policy_aliases: ClassVar[Dict[str, Type[BasePolicy]]] = {
        "QTable": QTable,
        "PolynomialApprox": PolynomialApproximation,
        "RBFApprox": RBFApproximation,
    }
    
    exploration_schedule: Schedule
    q_table: QTable
    q_table_target: QTable
    policy: QLPolicy
    
    
    def __init__(
        self,
        policy: Union[str, Type[QLearningPolicy]],
        env: Union[GymEnv, str],
        gamma: float = 0.99,        
        exploration_fraction: float = 0.1,
        exploration_initial_eps: float = 1.0,
        exploration_final_eps: float = 0.05,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[torch.device, str] = "auto",
        _init_setup_model: bool = True,
    ) -> None:
        super().__init__(
            policy,
            env,
            gamma=gamma,
            exploration_fraction=exploration_fraction,
            exploration_initial_eps=exploration_initial_eps,
            exploration_final_eps=exploration_final_eps,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            seed=seed,
            device=device,
            _init_setup_model=False,
        )
        self.exploration_initial_eps = exploration_initial_eps
        self.exploration_final_eps = exploration_final_eps
        self.exploration_fraction = exploration_fraction
        # For updating the target table
        self._n_calls = 0
        # "epsilon" for the epsilon-greedy exploration
        self.exploration_rate = 0.0
        
        if _init_setup_model:
            self._setup_model()
    
    def _setup_model(self) -> None:
        super()._setup_model()
            
            
        
