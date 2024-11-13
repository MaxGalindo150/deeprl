from abc import ABC, abstractmethod

from pathlib import Path
import torch


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
    def get_env(self):
        """
        Return the environment associated with the agent.

        **Abstract Method:** Must be implemented by subclasses to return 
        the environment object associated with the agent.
        
        :return: The environment object.
        :rtype: object
        """
        pass
    

    @staticmethod
    def _save_to_file(save_path, data=None, params=None, cloudpickle=False):
        """
        Save model data and parameters to a file, automatically adding a file extension if needed.

        :param save_path: (str or pathlib.Path) The base path to save the model to (without extension).
        :param data: (dict) Additional data to save.
        :param params: (dict) Model parameters to save.
        :param cloudpickle: (bool) Whether to use cloudpickle for serialization.
        """
        # Ensure save_path is a Path object and add default extension if not present
        save_path = Path(save_path)
        if save_path.suffix == "":
            save_path = save_path.with_suffix(".pth")  # Default extension

        # Save the data and parameters
        with open(save_path, "wb") as file:
            if cloudpickle:
                import cloudpickle
                cloudpickle.dump((data, params), file)
            else:
                import pickle
                pickle.dump((data, params), file)

        print(f"Model saved to {save_path}")

    
    @classmethod
    def load(cls, filepath):
        """
        Load an agent from a saved file.

        :param filepath: Path to the file to load the agent from (without extension).
        :return: An instance of the agent.
        """
        filepath = filepath + ".pth"
        state = torch.load(filepath)
        
        # Dynamically create the agent
        agent = cls(
            env=None,  # Placeholder, must be set after loading
            policy=None,  # Will be reconstructed
            reward_shaping=None  # Will be reconstructed
        )
        
        # Restore parameters
        agent.is_continuous = state["is_continuous"]
        if agent.is_continuous:
            agent.approximator.weights = torch.tensor(state["weights"])
        else:
            agent.q_table = torch.tensor(state["q_table"])
        agent.policy = state.get("policy") 
        agent.reward_shaping = state.get("reward_shaping")
        agent.learning_rate = state["learning_rate"]
        agent.discount_factor = state["discount_factor"]
        
        return agent
