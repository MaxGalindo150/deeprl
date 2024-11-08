from abc import ABC, abstractmethod

class Agent(ABC):
    """
    Base class for all reinforcement learning agents.

    This abstract class defines the core interface that all reinforcement 
    learning agents must implement, serving as a blueprint. It includes 
    essential methods for:
    - Selecting actions
    - Learning from experience
    - Saving and loading model parameters
    - Updating the policy
    - Interacting with the environment
    
    All subclasses must implement these methods to ensure consistent 
    functionality across different types of agents.
    """
    
    @abstractmethod
    def act(self, state):
        """
        Select an action based on the current state of the environment.

        **Abstract Method:** Must be implemented by subclasses to define 
        the agent's action-selection mechanism.
        
        :param state: The current state of the environment.
        :type state: object
        :return: The action chosen by the agent.
        :rtype: object
        """
        pass

    @abstractmethod
    def learn(self):
        """
        Update the agent's parameters based on experience.

        **Abstract Method:** Subclasses implement this method to define 
        the learning process, updating the agent’s parameters.
        
        :return: None
        """
        pass
    
    @abstractmethod
    def save(self, filepath):
        """
        Save the agent's parameters to a specified file.

        **Abstract Method:** Can be customized in subclasses to specify 
        the format and details of parameter saving.

        :param filepath: Path where parameters will be saved.
        :type filepath: str
        :return: None
        """
        pass
    
    @abstractmethod
    def load(self, filepath):
        """
        Load the agent's parameters from a specified file.

        **Abstract Method:** Can be customized in subclasses to specify 
        the format and details of parameter loading.

        :param filepath: Path from which parameters will be loaded.
        :type filepath: str
        :return: None
        """
        pass
    
    
    def update_policy(self):
        """
        Update the agent's policy based on current parameters.

        **Abstract Method:** Typically called after learning to adjust 
        the policy according to updated parameters.
        
        :return: None
        """
        pass

    @abstractmethod
    def interact(self, env, episodes=1):
        """
        Interact with the environment over a specified number of episodes.

        **Abstract Method:** Defines how the agent interacts with the 
        environment, typically used to gather experience or evaluate 
        performance.

        :param env: The environment to interact with.
        :type env: gymnasium.Env
        :param episodes: Number of episodes for interaction.
        :type episodes: int
        :return: A list of accumulated rewards per episode.
        :rtype: list
        """
        pass
