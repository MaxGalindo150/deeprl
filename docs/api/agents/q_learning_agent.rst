######################
QLearningAgent
######################

The ``QLearningAgent`` class implements the Q-learning algorithm, a model-free reinforcement learning method used to find the optimal policy for Markov Decision Processes (MDPs). This implementation supports a customizable exploration strategy and is compatible with PyTorch.

**Algorithm Overview**:

Q-learning is an off-policy algorithm that updates the action-value function, :math:`Q(s, a)`, using the following Bellman equation:

.. math::

    Q(s, a) = Q(s, a) + \alpha \left[ r + \gamma \max_a' Q(s', a') - Q(s, a) \right]

- :math:`s` is the current state.
- :math:`a` is the action taken in state :math:`s`.
- :math:`r` is the reward received after transitioning from :math:`s` to :math:`s'`.
- :math:`s'` is the next state.
- :math:`\alpha` is the learning rate.
- :math:`\gamma` is the discount factor.
- :math:`\max_a' Q(s', a')` is the maximum Q-value for the next state :math:`s'`.

The algorithm iteratively updates the Q-table for each state-action pair based on observed transitions, eventually converging to an optimal policy.

.. note::  
    Environments with sparse or uninformative reward signals, such as ``Frozen-Lake`` or ``Mountain-Car``, can hinder the agent's learning progress. To address this challenge, you can:

    - Utilize the ``step_penalty`` parameter to penalize excessive steps and encourage faster convergence.
    
    - Take advantage of the ``reward_shaping`` module to design custom reward signals that guide the agent toward desirable behaviors.
    
    - Explore alternative algorithms like DQN or PPO, which are better suited for environments with sparse rewards due to their ability to handle delayed feedback effectively.

    Reward shaping is particularly useful for improving exploration and speeding up learning in complex environments.


**Available Reward Shaping Techniques**:

The following reward shaping techniques are available in the `reward_shaping` module. Please refer to the individual documentation for more details and examples. There is reward shaping available for some specific environments.

.. autosummary::
   :nosignatures:

  deeprl.reward_shaping.step_penalty_shaping.StepPenaltyShaping
  deeprl.reward_shaping.potential_based_shaping.PotentialBasedShaping
  deeprl.reward_shaping.distance_based_shaping.DistanceBasedShaping
  deeprl.reward_shaping.mountain_car_reward_shaping.MountainCarRewardShaping


***********************
Features
***********************

- **Customizable Exploration Strategy**:
  The agent uses a policy-based exploration mechanism. By default, it employs an epsilon-greedy policy, but any compatible policy can be provided.

- **Training Visualization**:
  The agent supports progress tracking and training visualization.

- **Save and Load Q-Table**:
  The Q-table can be saved to a file and loaded later for testing or resuming training.

- **Performance Monitoring**:
  Verbose mode prints detailed training progress, including total and average rewards, steps, and epsilon values.

**Available Policies**:

.. autosummary::
   :nosignatures:

   deeprl.policies.epsilon_greedy_policy.EpsilonGreedyPolicy
   deeprl.policies.epsilon_greedy_decay_policy.EpsilonGreedyDecayPolicy

***********************
Example
***********************

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
Can I use?
***********************

.. list-table::
   :header-rows: 1

   * - Space
     - Action
     - Observation
   * - Discrete
     - ✅
     - ✅
   * - Box
     - ❌
     - ❌
   * - MultiDiscrete
     - ✅
     - ✅
   * - MultiBinary
     - ✅
     - ✅

.. note::
    The Q-learning agent supports only environments with discrete action and observation spaces. For environments with continuous spaces, you can:
    
    - Apply discretization techniques to transform continuous spaces into discrete representations.
    
    - Leverage the built-in function approximation methods, which enable the agent to generalize across continuous state or action spaces by using approximators.

    - For non-linear function approximation, consider using deep reinforcement learning algorithms like DQN or PPO, which are well-suited for complex environments with continuous spaces. (future implementation)

    Function approximation significantly expands the applicability of Q-learning, making it suitable for complex environments where discretization is infeasible or inefficient.


**Available Function Approximations**:

The following function approximators are available in the `function_approximations` module. Please refer to the individual documentation for more details and examples.

.. autosummary::
   :nosignatures:

   deeprl.function_approximations.polynomial_basis_approximator.PolynomialBasisApproximator
   deeprl.function_approximations.radial_basis_approximator.RBFBasisApproximator

***********************
Parameters
***********************

.. autoclass:: deeprl.agents.q_learning_agent.QLearningAgent
   :members:
   :inherited-members:

***********************
See Also
***********************

- :class:`~deeprl.agents.policy_iteration_agent.PolicyIterationAgent` for a related algorithm that alternates explicitly between policy evaluation and policy improvement.
- :class:`~deeprl.agents.value_iteration_agent.ValueIterationAgent` for an alternative dynamic programming method that uses value iteration to find the optimal policy.