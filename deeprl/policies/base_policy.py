from abc import ABC, abstractmethod

class BasePolicy(ABC):
    """
    Abstract base class for DeepRL-compatible custom policies.
    
    This class defines the interface that all custom policies must implement.
    """

    @abstractmethod
    def select_action(self, state, *args, **kwargs):
        """
        Selects an action based on the given state and additional parameters.
        
        :param state: The current state.
        :param args: Additional positional arguments.
        :param kwargs: Additional keyword arguments.
        :return: The selected action.
        """
        pass

    
    @abstractmethod
    def update(self, **kwargs):
        """
        Updates the policy based on provided keyword arguments (optional).
        
        :param kwargs: Additional keyword arguments.
        """
        pass
