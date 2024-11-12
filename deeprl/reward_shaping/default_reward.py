from deeprl.reward_shaping.base_reward_shaping import BaseRewardShaping

class DefaultReward(BaseRewardShaping):
    
    def shape(self, state, action, next_state, reward):
        return reward