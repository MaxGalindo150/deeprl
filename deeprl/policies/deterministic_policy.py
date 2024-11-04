from deeprl.policies.base_policy import BasePolicy

class DeterministicPolicy(BasePolicy):
    """
    Determinisc policy implementation.
    """
    
    def __init__(self, policy_table):
        """
        Initialize the policy with a given policy table.
        
        :param policy_table: Dictionary or array-like object where policy_table[state] is the action to take in state state.
        """
        
        self.policy_table = policy_table
        
    def select_action(self, state, *args, **kwargs):
        """
        Select an action using the deterministic policy.
        
        param state: The state.
        return: The action to take in the given state.
        """
        
        return self.policy_table[state]