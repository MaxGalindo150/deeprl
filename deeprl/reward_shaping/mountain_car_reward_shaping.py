from deeprl.reward_shaping.base_reward_shaping import BaseRewardShaping

class MountainCarRewardShaping(BaseRewardShaping):
    """
    Reward shaping class for the MountainCar environment.

    This class implements custom reward shaping to guide the agent
    in the MountainCar environment by providing progress incentives 
    and a bonus for reaching the goal.
    
    :param state: Current state as a tuple (position, velocity).
    :param action: Action taken by the agent (optional for this function).
    :param next_state: Next state as a tuple (position, velocity).
    :param reward: Original reward from the environment.
    :return: Shaped reward.
    """
    def shape(self, state, action, next_state, reward):
        """
        Apply custom reward shaping.
        """
        position, velocity = next_state
        reward += 10 * (position - state[0])  # Bonus for advancing towards the goal
        if position >= 0.5:  # Goal reached
            reward += 100
        return reward