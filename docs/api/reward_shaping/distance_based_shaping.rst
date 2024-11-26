#########################
DistanceBasedShaping
#########################

The `DistanceBasedShaping` class implements a reward shaping strategy based on the distance to a predefined goal. This approach adjusts the reward signal by incorporating the proximity of the agent's state to the goal, encouraging behaviors that reduce the distance to the target.

************************
How It Works
************************

The `DistanceBasedShaping` class modifies the reward by calculating the negative Euclidean distance between the next state and the goal state. The shaped reward combines the original reward with this distance-based adjustment, incentivizing the agent to move closer to the goal state.

Key functionalities include:
- **Goal State Definition**: The user specifies a target goal state during initialization.
- **Dynamic Reward Adjustment**: The reward is modified at each step based on the agent's proximity to the goal.

This strategy is particularly effective in navigation tasks or environments where reaching a specific state is critical to success.

***********************
Example
***********************

Here’s how to use the `DistanceBasedShaping` class:

.. code-block:: python

    import numpy as np
    from deeprl.reward_shaping import DistanceBasedShaping

    # Define the goal state
    goal_state = np.array([0.0, 0.0])

    # Initialize the reward shaping strategy
    reward_shaping = DistanceBasedShaping(goal_state)

    # Example interaction
    state = np.array([1.0, 1.0])
    next_state = np.array([0.5, 0.5])
    action = None  # Action is not used in this shaping strategy
    reward = 10.0

    # Compute the shaped reward
    shaped_reward = reward_shaping.shape(state, action, next_state, reward)
    print("Shaped Reward:", shaped_reward)

***********************
Parameters
***********************

.. autoclass:: deeprl.reward_shaping.distance_based_shaping.DistanceBasedShaping
   :members:

***********************
See Also
***********************

- :class:`~deeprl.reward_shaping.base_reward_shaping.BaseRewardShaping` for the abstract base class that `DistanceBasedShaping` inherits from.
- :class:`~deeprl.reward_shaping.potential_based_shaping.PotentialBasedShaping` for a potential-based reward shaping strategy.
- :class:`~deeprl.reward_shaping.step_penalty_shaping.StepPenaltyShaping` for a step penalty-based reward shaping strategy.
- :class:`~deeprl.reward_shaping.mountain_car_reward_shaping.MountainCarRewardShaping` for a reward shaping strategy tailored to the MountainCar environment.

***********************
References
***********************

1. Ng, A. Y., Harada, D., & Russell, S. J. (1999). *Policy Invariance Under Reward Transformations: Theory and Application to Reward Shaping*. In ICML.
2. Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction (2nd ed.)*. MIT Press. Available at: http://incompleteideas.net/book/the-book-2nd.html
3. Mataric, M. J. (1994). *Reward Functions for Accelerated Learning*. In ICML.
