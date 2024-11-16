import warnings
from typing import Any, ClassVar, Dict, List, Optional, Tuple, Type, TypeVar, Union

from deeprl.common.off_policy_agent import OffPolicyAgent

SelfQLearning = TypeVar("SelfQLearning", bound="QL")

class QL(OffPolicyAgent):
    """
    Q-Learning agent.
    
    Reference: Sutton and Barto, Reinforcement Learning: An Introduction, 2nd edition, 2018.
    
    :param policy: The policy model to use (e.g. Q-table)
    :param env: The environment to learn from (if registered in gym, can be str)
    :param gamma: The discount factor
    :param epsiolon: The exploration rate
    :param exploration_initial_eps: Initial value of random action probability
    :param exploration_fraction: Fraction of entire training period over which the exploration rate is reduced
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
    policy: QLPolicy
