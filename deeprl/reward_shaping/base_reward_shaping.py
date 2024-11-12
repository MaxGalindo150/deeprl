from abc import ABC, abstractmethod

class BaseRewardShaping(ABC):
    """
    Abstract base class for reward shaping strategies.
    """

    @abstractmethod
    def shape(self, state, action, next_state, reward):
        """
        Shape the reward based on the provided parameters.

        :param state: Current state.
        :param action: Action taken.
        :param next_state: State after taking the action.
        :param reward: Original reward from the environment.
        :return: Shaped reward.
        """
        pass
