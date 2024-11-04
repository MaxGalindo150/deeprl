import torch
from deeprl.policies.base_policy import BasePolicy

class EpsilonGreedyPolicy(BasePolicy):
    """
    Epsilon-Greedy policy.
    """
    
    def __init__(self, epsilon=0.1):
        """"
        Initialize the policy with a given epsilon value.
        
        :param epsilon: The epsilon value.
        """
        self.epsilon = epsilon
        
    def select_action(self, state, q_values):
        """"
        Select an action using the epsilon-greedy policy.
        
        :param state: The state.
        :param q_values: List or array of Q-values for each posible action.
        """
        
        if torch.rand(1).item() < self.epsilon:
            # Exploration: Select a random action
            return torch.randint(len(q_values), (1,)).item()
        else:
            # Exploitation: Select the action with the highest Q-value
            return torch.argmax(q_values).item()