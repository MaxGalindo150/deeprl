import numpy as np
from deeprl.reward_shaping.base_reward_shaping import BaseRewardShaping

class DistanceBasedShaping(BaseRewardShaping):
    """
    Reward shaping based on the distance to a goal.
    """

    def __init__(self, goal_state):
        """
        Initialize with the goal state.

        :param goal_state: Target state to calculate distance from.
        """
        self.goal_state = np.array(goal_state)

    def shape(self, state, action, next_state, reward):
        """
        Modify the reward based on the distance to the goal.

        :param state: Current state.
        :param action: Action taken.
        :param next_state: State after taking the action.
        :param reward: Original reward from the environment.
        :return: Shaped reward.
        """
        distance_to_goal = -np.linalg.norm(next_state - self.goal_state)
        return reward + distance_to_goal
