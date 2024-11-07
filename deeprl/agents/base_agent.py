from abc import ABC, abstractmethod

class Agent(ABC):
    """
    Base class for all reinforcement learning agents.

    This class defines the basic interface that all reinforcement learning 
    agents must implement. It includes methods for selecting actions, 
    learning from experience, saving and loading parameters, and updating 
    the policy.
    """
    
    @abstractmethod
    def act(self, state):
        """
        Select an action based on the current state of the environment.

        This method should be implemented by all subclasses to define how 
        the agent selects actions.

        :param state: The current state of the environment.
        :type state: object
        :return: The selected action.
        :rtype: object
        """
        pass

    @abstractmethod
    def learn(self):
        """
        Update the agent's parameters based on the collected experience.

        This method should be implemented by all subclasses to define how 
        the agent learns from experience.
        
        :return: None
        """
        pass
    
    @abstractmethod
    def save(self, filepath):
        """
        Save the agent's parameters to a file.

        This method can be overridden by subclasses to define how the 
        agent's parameters are saved.

        :param filepath: The path to the file where the parameters will 
                         be saved.
        :type filepath: str
        :return: None
        """
        pass
    
    @abstractmethod
    def load(self, filepath):
        """
        Load the agent's parameters from a file.

        This method can be overridden by subclasses to define how the 
        agent's parameters are loaded.

        :param filepath: The path to the file from which the parameters 
                         will be loaded.
        :type filepath: str
        :return: None
        """
        pass
    
    @abstractmethod
    def update_policy(self):
        """
        Update the policy based on the agent's parameters.

        This method can be overridden by subclasses to define how the 
        agent's policy is updated.
        
        :return: None
        """
        pass

    @abstractmethod
    def interact(self, env, episodes=1):
        """
        Interactúa con el entorno durante un número específico de episodios.

        :param env: El entorno con el que interactuar.
        :type env: gymnasium.Env
        :param episodes: Número de episodios para la interacción.
        :type episodes: int
        :return: Lista de recompensas acumuladas por episodio.
        :rtype: list
        """
        pass
