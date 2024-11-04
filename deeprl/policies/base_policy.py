from abc import ABC, abstractmethod

class BasePolicy(ABC):
    """"
    Abstract base class for DeepRL-compatible custom policies.
    """
    
    @abstractmethod
    def select_action(self, state):
        """
        Selects an action based on the given state.
        
        :param state: State to select an action for.
        :return: Selected action.
        """
        pass
    
    