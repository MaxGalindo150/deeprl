PolicyIterationAgent
=====================
The `PolicyIterationAgent` class implements the policy iteration algorithm, a dynamic programming method used for solving Markov Decision Processes.

.. autoclass:: deeprl.agents.policy_iteration_agent.PolicyIterationAgent
   :members:
   :undoc-members:
   :show-inheritance:

**Example Usage**:
Here's how to use `PolicyIterationAgent`:

.. code-block:: python

    from deeprl.agents.policy_iteration_agent import PolicyIterationAgent
    from deeprl.environments import GymnasiumEnvWrapper
    import gymnasium as gym

    # Create and wrap an environment
    env = GymnasiumEnvWrapper('FrozenLake-v1')
    agent = PolicyIterationAgent(env)

    # Train the agent
    agent.learn()

**Method Summary**:
- `learn(self)`: Implements the policy iteration algorithm to determine the optimal policy.

**Details**:
The `PolicyIterationAgent` alternates between policy evaluation and policy improvement steps until convergence.

**Note**: This agent is designed to work with environments that support discrete action and state spaces. The `GymnasiumEnvWrapper` class can be used to wrap Gymnasium environments for compatibility with DeepRL.

**See Also**: 
- `ValueIterationAgent` for an alternative dynamic programming method that uses value iteration to find the optimal policy.
- `BaseEnvironment` for creating custom environments that can be used with DeepRL algorithms.
- `GymnasiumEnvWrapper` for integrating Gymnasium environments with DeepRL.