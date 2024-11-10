#######################
BaseRewardShaping
#######################

The ``BaseRewardShaping`` class serves as an abstract base class for implementing reward shaping strategies in reinforcement learning. Reward shaping allows you to modify the reward signal to guide the agent toward desired behaviors, especially in environments with sparse or uninformative rewards.

************************
How It Works
************************

The ``BaseRewardShaping`` class provides an abstract interface for defining custom reward shaping mechanisms. Subclasses must implement the ``shape`` method, which modifies the original reward based on the current state, action, next state, and the reward received from the environment.

Reward shaping can be useful in scenarios where:
- The environment's original reward signal is sparse, making it difficult for the agent to learn.
- Additional guidance is needed to encourage exploration or specific behaviors.

By extending this class, users can create tailored reward shaping strategies that fit their specific needs.

***********************
Example
***********************

Here’s how to implement and use a custom reward shaping strategy:

.. code-block:: python

    from deeprl.reward_shaping import BaseRewardShaping

    class DistanceRewardShaping(BaseRewardShaping):
        def shape(self, state, action, next_state, reward):
            # Example: Add a small bonus based on proximity to the goal
            distance_to_goal = next_state[0]  # Assuming next_state[0] represents distance
            shaped_reward = reward + (1.0 / (1.0 + distance_to_goal))
            return shaped_reward

    # Example usage
    reward_shaping = DistanceRewardShaping()
    shaped_reward = reward_shaping.shape(state, action, next_state, reward)

***********************
Parameters
***********************

.. autoclass:: deeprl.reward_shaping.base_reward_shaping.BaseRewardShaping
   :members:

***********************
See Also
***********************

- :class:`~deeprl.reward_shaping.potential_based_shaping.PotentialBasedShaping` for a potential-based reward shaping strategy.
- :class:`~deeprl.reward_shaping.step_penalty.StepPenaltyShaping` for penalizing the agent based on the number of steps taken.

***********************
References
***********************

1. Ng, A. Y., Harada, D., & Russell, S. J. (1999). *Policy Invariance Under Reward Transformations: Theory and Application to Reward Shaping*. In ICML.
2. Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction (2nd ed.)*. MIT Press. Available at: http://incompleteideas.net/book/the-book-2nd.html
3. Mataric, M. J. (1994). *Reward Functions for Accelerated Learning*. In ICML.
