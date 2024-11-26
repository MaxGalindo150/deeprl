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

    import gymnasium as gym
    from deeprl.reward_shaping import MountainCarRewardShaping

    # Initialize the MountainCar environment
    env = gym.make('MountainCar-v0')

    # Initialize the reward shaping strategy
    reward_shaping = MountainCarRewardShaping()

    # Example interaction
    state = (-0.5, 0.0)  # Position, velocity
    next_state = (-0.4, 0.1)  # Position, velocity
    action = 0  # Example action
    reward = -1.0  # Original reward from the environment

    # Compute the shaped reward
    shaped_reward = reward_shaping.shape(state, action, next_state, reward)
    print("Shaped Reward:", shaped_reward)

***********************
Parameters
***********************

.. autoclass:: deeprl.reward_shaping.mountain_car_reward_shaping.MountainCarRewardShaping
   :members:

***********************
See Also
***********************

- :class:`~deeprl.reward_shaping.base_reward_shaping.BaseRewardShaping` for the abstract base class that `MountainCarRewardShaping` inherits from.
- :class:`~deeprl.reward_shaping.potential_based_shaping.PotentialBasedShaping` for a potential-based reward shaping strategy.
- :class:`~deeprl.reward_shaping.step_penalty_shaping.StepPenaltyShaping` for a step penalty-based reward shaping strategy.
- :class:`~deeprl.reward_shaping.distance_based_shaping.DistanceBasedShaping` for a reward shaping strategy based on proximity to a goal.

***********************
References
***********************

1. Ng, A. Y., Harada, D., & Russell, S. J. (1999). *Policy Invariance Under Reward Transformations: Theory and Application to Reward Shaping*. In ICML.
2. Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction (2nd ed.)*. MIT Press. Available at: http://incompleteideas.net/book/the-book-2nd.html
3. Mataric, M. J. (1994). *Reward Functions for Accelerated Learning*. In ICML.
