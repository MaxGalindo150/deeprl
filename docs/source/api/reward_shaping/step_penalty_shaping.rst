#########################
StepPenaltyShaping
#########################

The `StepPenaltyShaping` class implements a default reward shaping strategy that applies a step penalty. This is useful for environments with sparse rewards or when a penalty for taking steps without achieving a reward is desired. By discouraging excessive steps, this shaping strategy can help the agent converge to a solution more efficiently.

************************
How It Works
************************

The `StepPenaltyShaping` class modifies the reward signal by subtracting a penalty for each step where the agent does not receive an explicit reward. The step penalty is user-defined and can be adjusted based on the desired level of discouragement for unnecessary actions.

Key functionalities include:
- **Step Penalty**: Subtracts a fixed penalty from the reward signal if the reward is zero, guiding the agent to minimize unnecessary steps.
- **Default Behavior**: If no penalty is specified, the class behaves as a pass-through, leaving the reward unchanged.

This strategy is particularly useful in environments where the agent receives sparse rewards, such as `Frozen Lake` or `Mountain Car`.

***********************
Example
***********************

Here’s how to use the `StepPenaltyShaping` class:

.. code-block:: python

    from deeprl.environments import GymnasiumEnvWrapper
    from deeprl.agents.q_learning_agent import QLearningAgent
    from deeprl.policies.epsilon_greedy_decay_policy import EpsilonGreedyDecayPolicy
    from deeprl.reward_shaping.step_penalty_shaping import StepPenaltyShaping

    def main():
        
        # Configure the FrozenLake environment
        env = GymnasiumEnvWrapper('FrozenLake-v1', is_slippery=False)
        
        # Initialize the agent with a decaying epsilon-greedy policy
        policy = EpsilonGreedyDecayPolicy(epsilon=1, decay_rate=0.99, min_epsilon=0.1)
        
        agent = QLearningAgent(
            env=env, 
            policy=policy,
            reward_shaping=StepPenaltyShaping(step_penalty=-0.1),
            verbose=True
        )
        
        # Train the agent
        agent.learn(episodes=100000, max_steps=10000, save_train_graph=True)
        
        # Evaluate the agent
        rewards = agent.interact(episodes=10, render=True, save_test_graph=True)

    if __name__ == '__main__':
        main()

***********************
Parameters
***********************

.. autoclass:: deeprl.reward_shaping.step_penalty_shaping.StepPenaltyShaping
   :members:

***********************
See Also
***********************

- :class:`~deeprl.reward_shaping.base_reward_shaping.BaseRewardShaping` for the abstract base class that `StepPenaltyShaping` inherits from.
- :class:`~deeprl.reward_shaping.distance_based_shaping.DistanceBasedShaping` for a distance-based reward shaping strategy.
- :class:`~deeprl.reward_shaping.potential_based_shaping.PotentialBasedShaping` for a shaping strategy based on potential functions.

***********************
References
***********************

1. Ng, A. Y., Harada, D., & Russell, S. J. (1999). *Policy Invariance Under Reward Transformations: Theory and Application to Reward Shaping*. In ICML.
2. Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction (2nd ed.)*. MIT Press. Available at: http://incompleteideas.net/book/the-book-2nd.html
3. Mataric, M. J. (1994). *Reward Functions for Accelerated Learning*. In ICML.
