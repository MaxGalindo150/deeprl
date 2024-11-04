from abc import ABC, abstractmethod

class Agent(ABC):
    """Base class for all reinforcement learning agents."""
    
    @abstractmethod
    def act(self, state):
        """
        Select an action based on the state.
        
        :param state: The current state of the environment.
        :return: The selected action.
        """
        pass

    @abstractmethod
    def learn(self):
        """"
        Update the agent's parameters based on the collected experience.
        """
        pass
    
    def save(self, filepath):
        """
        Save the agent's parameters to a file.
        
        :param filepath: The path to the file.
        :return: None
        """
        pass
    
    def load(self, filepath):
        """
        Load the agent's parameters from a file.
        
        :param filepath: The path to the file.
        :return: None
        """
        pass
    
    def update_policy(self):
        """
        Update the policy based on the agent's parameters.
        
        :return: None
        """