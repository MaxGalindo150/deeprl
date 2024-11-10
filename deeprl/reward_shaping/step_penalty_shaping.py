from deeprl.reward_shaping.base_reward_shaping import BaseRewardShaping

class StepPenaltyShaping(BaseRewardShaping):
    """
    Default reward shaping that applies a step penalty if provided.
    This is useful for environments with sparse rewards or when a 
    penalty for steps is desired.
    """

    def __init__(self, step_penalty=0.0):
        """
        Initialize the default reward shaping with a step penalty.

        :param step_penalty: Penalty to apply for each step without reward.
        """
        self.step_penalty = step_penalty

    def shape(self, state, action, next_state, reward):
        """
        Apply default reward shaping logic.

        :param state: Current state (not used in this shaping logic).
        :param action: Action taken by the agent (not used in this shaping logic).
        :param next_state: Next state (not used in this shaping logic).
        :param reward: Original reward from the environment.
        :return: Shaped reward with step penalty applied.
        """
        if reward == 0 and self.step_penalty != 0:
            return reward + self.step_penalty
        return reward
