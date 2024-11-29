# import numpy as np
# from typing import Any, Dict, Optional, Tuple, Type, Union
# from gymnasium import spaces
# from deeprl.common.policies import BasePolicy


# class TabularPolicy(BasePolicy):
#     """
#     A tabular policy for discrete state and action spaces, storing the Q-values explicitly.

#     :param observation_space: The observation space of the environment
#     :param action_space: The action space of the environment
#     :param learning_rate: Learning rate for updating Q-values
#     :param gamma: Discount factor for future rewards
#     :param epsilon: Exploration rate for epsilon-greedy policy
#     :param optimizer_class: Ignored for tabular policies
#     :param optimizer_kwargs: Ignored for tabular policies
#     """

#     def __init__(
#         self,
#         observation_space: spaces.Space,
#         action_space: spaces.Space,
#         learning_rate: float = 0.1,
#         gamma: float = 0.99,
#         epsilon: float = 1.0,
#         features_extractor_class: Optional[Type] = None,  # Not used for tabular
#         features_extractor_kwargs: Optional[Dict[str, Any]] = None,  # Not used for tabular
#         normalize_images: bool = False,
#         optimizer_class: Optional[Type] = None,  # Not used for tabular
#         optimizer_kwargs: Optional[Dict[str, Any]] = None,  # Not used for tabular
#     ):
#         super().__init__(
#             observation_space=observation_space,
#             action_space=action_space,
#             features_extractor_class=features_extractor_class,
#             features_extractor_kwargs=features_extractor_kwargs,
#             normalize_images=normalize_images,
#         )
#         # Ensure observation and action spaces are discrete
#         assert isinstance(observation_space, spaces.Discrete), "TabularPolicy requires discrete observation space"
#         assert isinstance(action_space, spaces.Discrete), "TabularPolicy requires discrete action space"

#         self.learning_rate = learning_rate
#         self.gamma = gamma
#         self.epsilon = epsilon

#         # Initialize the Q-table with zeros
#         self.q_table = np.zeros((observation_space.n, action_space.n))

#     def _predict(self, observation: np.ndarray, deterministic: bool = False) -> np.ndarray:
#         """
#         Select an action based on the policy (epsilon-greedy).

#         :param observation: Current state (integer index).
#         :param deterministic: If True, select the action with the highest Q-value.
#         :return: Selected action.
#         """
#         if not deterministic and np.random.rand() < self.epsilon:
#             # Explore: choose a random action
#             return np.array(self.action_space.sample())
#         else:
#             # Exploit: choose the best action
#             return np.array(np.argmax(self.q_table[observation]))

#     def update(self, state: int, action: int, reward: float, next_state: int, done: bool) -> None:
#         """
#         Update the Q-value for the given state-action pair using the Q-learning update rule.

#         :param state: Current state (integer index).
#         :param action: Action taken.
#         :param reward: Reward received.
#         :param next_state: Next state (integer index).
#         :param done: Whether the episode has terminated.
#         """
#         target = reward
#         if not done:
#             target += self.gamma * np.max(self.q_table[next_state])
#         td_error = target - self.q_table[state, action]
#         self.q_table[state, action] += self.learning_rate * td_error

#     def reset_q_table(self) -> None:
#         """Reset the Q-table to all zeros."""
#         self.q_table = np.zeros_like(self.q_table)

#     def set_epsilon(self, epsilon: float) -> None:
#         """
#         Set the exploration rate.

#         :param epsilon: New epsilon value.
#         """
#         self.epsilon = epsilon
