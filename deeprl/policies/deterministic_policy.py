import torch
from deeprl.policies.base_policy import BasePolicy

class DeterministicPolicy(BasePolicy):
    """
    Deterministic policy implementation.
    """
    
    def __init__(self, observation_space):
        """
        Initialize the policy.
        
        :param observation_space: The observation space.
        """
        self.policy_table = torch.zeros(observation_space.n)
    
    def select_action(self, state, *args, **kwargs):
        """
        Select an action using the deterministic policy.
        
        :param state: The state.
        :return: The action to take in the given state.
        """
        try:
            return self.policy_table[state].item()
        except IndexError:
            raise IndexError(f"State {state} is out of range for the policy table.")
        except KeyError:
            raise KeyError(f"State {state} not found in policy_table.")
        
    
    def update_policy(self, state, value):
        """
        Update the policy with the given state and value.
        
        :param state: The state.
        :param value: The value.
        """
        try:
            self.policy_table[state] = value
        except IndexError:
            raise IndexError(f"State {state} is out of range for the policy table.")