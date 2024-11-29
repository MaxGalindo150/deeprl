import numpy as np
from typing import Any, Dict, Optional, Tuple, Union
from gymnasium import spaces

from deeprl.common.policies import TabularModel

class QTable(TabularModel):
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
        super().__init__(observation_space, action_space)

        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon

        # Initialize Q-table with zeros
        self.table = np.zeros((observation_space.n, action_space.n))

    def predict(self, state: int, deterministic: bool = False) -> int:
        """
        Predict the next action using the Q-Table.

        :param state: Current state as an integer index.
        :param deterministic: Whether to select the action deterministically.
        :return: Selected action as an integer.
        """
        if deterministic or np.random.rand() >= self.epsilon:
            return np.argmax(self.table[state])
        else:
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
            target += self.gamma * np.max(self.table[next_state])

        self.table[state, action] += self.learning_rate * (target - self.table[state, action])

    def reset(self) -> None:
        """
        Reset the Q-table to all zeros.
        """
        self.table.fill(0)