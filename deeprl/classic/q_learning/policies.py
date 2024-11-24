import numpy as np
from typing import Any, Dict, Optional, Tuple, Union
from gymnasium import spaces

from deeprl.common.policies import BaseTabularPolicy

class QTable(BaseTabularPolicy):
    """
    A Q-Table implementation for tabular reinforcement learning.

    :param observation_space: The observation space of the environment.
    :param action_space: The action space of the environment.
    :param learning_rate: The learning rate for updating Q-values.
    :param gamma: Discount factor for future rewards.
    :param epsilon: Exploration rate for epsilon-greedy policy.
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        learning_rate: float = 0.1,
        gamma: float = 0.99,
        epsilon: float = 1.0,
    ):
        super().__init__(observation_space=observation_space, action_space=action_space)
        
        # Validate that spaces are discrete
        assert isinstance(observation_space, spaces.Discrete), "QTable only supports discrete observation spaces."
        assert isinstance(action_space, spaces.Discrete), "QTable only supports discrete action spaces."
        
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        
        # Initialize Q-table with zeros
        self.q_table = np.zeros((observation_space.n, action_space.n))

    def _predict(self, observation: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """
        Predict the best action using the Q-Table.

        :param observation: Current state (discrete index).
        :param deterministic: If True, select the action with the highest Q-value.
        :return: Selected action.
        """
        if deterministic or np.random.rand() >= self.epsilon:
            # Exploitation: Select the action with the highest Q-value
            return np.argmax(self.q_table[observation])
        else:
            # Exploration: Random action
            return self.action_space.sample()

    def update(self, state: int, action: int, reward: float, next_state: int, done: bool) -> None:
        """
        Update the Q-value for the given state-action pair using the Q-Learning update rule.

        :param state: Current state (integer index).
        :param action: Action taken.
        :param reward: Reward received.
        :param next_state: Next state (integer index).
        :param done: Whether the episode has terminated.
        """
        target = reward
        if not done:
            target += self.gamma * np.max(self.q_table[next_state])
        
        # Q-Learning update rule
        self.q_table[state, action] += self.learning_rate * (target - self.q_table[state, action])

    def set_epsilon(self, epsilon: float) -> None:
        """
        Set the exploration rate.

        :param epsilon: New epsilon value.
        """
        self.epsilon = epsilon

    def reset(self) -> None:
        """
        Reset the Q-table to all zeros.
        """
        self.q_table.fill(0)
