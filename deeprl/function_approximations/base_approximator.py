import torch
from abc import ABC, abstractmethod

class BaseApproximator(ABC):
    """
    Base class for function approximators in reinforcement learning.
    """

    @abstractmethod
    def compute_features(self, state):
        """
        Compute the feature vector for a given state.
        :param state: The state to compute features for.
        :return: A feature vector as a PyTorch tensor.
        """
        pass

    @abstractmethod
    def predict(self, state):
        """
        Predict the value for a given state.
        :param state: The state to predict the value for.
        :return: The predicted value as a PyTorch tensor.
        """
        pass

    @abstractmethod
    def update(self, state, target, alpha=0.1):
        """
        Update the approximator's parameters.
        :param state: The state to update.
        :param target: The target value for the update.
        :param alpha: Learning rate for the update.
        """
        pass
