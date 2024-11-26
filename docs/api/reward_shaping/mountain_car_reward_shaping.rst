################################
MountainCarRewardShaping
################################

The `MountainCarRewardShaping` class provides a custom reward shaping strategy tailored for the `MountainCar` environment. This shaping mechanism guides the agent by rewarding progress toward the goal and providing a significant bonus for reaching the target position.

************************
How It Works
************************

The `MountainCarRewardShaping` class modifies the reward signal to incentivize the agent to move towards the goal position (`position >= 0.5`). The shaping logic works as follows:

1. **Progress Incentive**:
   - A reward proportional to the change in the car's position is added, encouraging the agent to make forward progress.

2. **Goal Bonus**:
   - An additional reward of `+100` is provided when the agent reaches or exceeds the goal position.

This shaping strategy is specifically designed to accelerate learning in the `MountainCar` environment, where the original sparse reward signal can make it challenging for agents to learn effectively.

***********************
Example
***********************

Here’s how to use the `MountainCarRewardShaping` class:

.. code-block:: python

   from deeprl.environments import GymnasiumEnvWrapper
   from deeprl.agents.q_learning_agent import QLearningAgent
   from deeprl.function_approximations import RBFBasisApproximator
   from deeprl.reward_shaping import MountainCarRewardShaping

   def main():
      
      # Initialize the environment and approximator
      env = GymnasiumEnvWrapper('MountainCar-v0')
      approximator = RBFBasisApproximator(env=env, gamma=0.5, n_components=500)
         
      agent = QLearningAgent(
         env=env,
         learning_rate=0.1,
         discount_factor=0.99,
         is_continuous=True,
         approximator=approximator,
         reward_shaping=MountainCarRewardShaping(),
         verbose=True
      )
      
      # Train the agent
      agent.learn(episodes=10000, max_steps=10000, save_train_graph=True)
      
      # Evaluate the agent
      rewards = agent.interact(episodes=10, render=True, save_test_graph=True)

   if __name__ == '__main__':
      main()

***********************
Parameters
***********************

.. autoclass:: deeprl.reward_shaping.mountain_car_reward_shaping.MountainCarRewardShaping
   :members:
   :noindex:

***********************
See Also
***********************

- :class:`~deeprl.reward_shaping.base_reward_shaping.BaseRewardShaping` for the abstract base class that `MountainCarRewardShaping` inherits from.
- :class:`~deeprl.reward_shaping.step_penalty_shaping.StepPenaltyShaping` for a step penalty-based reward shaping strategy.
- :class:`~deeprl.reward_shaping.distance_based_shaping.DistanceBasedShaping` for a reward shaping strategy based on proximity to a goal.
- :class:`~deeprl.reward_shaping.potential_based_shaping.PotentialBasedShaping` for a potential-based reward shaping strategy.

***********************
References
***********************

1. Ng, A. Y., Harada, D., & Russell, S. J. (1999). *Policy Invariance Under Reward Transformations: Theory and Application to Reward Shaping*. In ICML.
2. Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction (2nd ed.)*. MIT Press. Available at: http://incompleteideas.net/book/the-book-2nd.html
3. Mataric, M. J. (1994). *Reward Functions for Accelerated Learning*. In ICML.