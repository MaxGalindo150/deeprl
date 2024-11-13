import torch
import numpy as np
from deeprl.common.base_policy import BasePolicy

class EpsilonGreedyPolicy(BasePolicy):
    """
    Epsilon-greedy policy for exploration and exploitation.
    
    :param epsilon: Probability of choosing a random action.
    :param decay: Whether to decay epsilon over time.
    :param decay_rate: Rate of epsilon decay.
    """

    def __init__(self, epsilon=0.1):
        """
        Initialize the epsilon-greedy policy.

        :param epsilon: Probability of choosing a random action.
        """
        self.epsilon = epsilon
    
    def select_action(self, q_values):
        """
        Select an action based on epsilon-greedy policy.

        :param q_values: Tensor of Q-values for the current state.
        :return: Selected action as an integer.
        """
        if torch.rand(1).item() < self.epsilon:
            # Explore: choose a random action
            return np.random.choice(len(q_values))
        else:
            # Exploit: choose the action with the highest Q-value
            return torch.argmax(q_values).item()

    def set_epsilon(self, epsilon):
        """
        Update epsilon for the policy.

        :param epsilon: New epsilon value.
        """
        self.epsilon = epsilon
        
    def update(self):
        
        pass