import torch
from deeprl.common.base_policy import BasePolicy

class SoftmaxPolicy(BasePolicy):
    """"
    Softmax policy implementation.
    """
    
    def __init__(self, temperature=1.0):
        """
        Initialize the policy with a given temperature value.
        
        :param temperature: Controls the level of exploration.
        """
        
        self.temperature = temperature
        
    def select_action(self, state, q_values):
        """
        Select an action using the softmax policy.
        
        :param state: The state.
        :param q_values: List or array of Q-values for each possible action.
        """
        
        exp_q = torch.exp(q_values / self.temperature)
        probabilities = exp_q / torch.sum(exp_q)
        return torch.multinomial(probabilities, 1).item()