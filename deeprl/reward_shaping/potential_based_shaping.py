from deeprl.reward_shaping.base_reward_shaping import BaseRewardShaping

class PotentialBasedShaping(BaseRewardShaping):
    """
    Reward shaping using potential-based shaping functions.
    """

    def __init__(self, potential_function, discount_factor=0.99):
        """
        Initialize with a potential function.

        :param potential_function: Callable that computes potential for a given state.
        :param discount_factor: Discount factor for shaping.
        """
        self.potential_function = potential_function
        self.discount_factor = discount_factor

    def shape(self, state, action, next_state, reward):
        """
        Modify the reward using potential shaping.

        :param state: Current state.
        :param action: Action taken.
        :param next_state: State after taking the action.
        :param reward: Original reward from the environment.
        :return: Shaped reward.
        """
        potential_current = self.potential_function(state)
        potential_next = self.potential_function(next_state)
        shaping_term = self.discount_factor * potential_next - potential_current
        return reward + shaping_term
