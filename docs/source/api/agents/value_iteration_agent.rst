ValueIterationAgent
====================
The `ValueIterationAgent` class implements the value iteration algorithm, a popular dynamic programming method for finding optimal policies in reinforcement learning.

.. autoclass:: deeprl.agents.value_iteration_agent.ValueIterationAgent
   :members:
   :undoc-members:
   :show-inheritance:

**Example Usage**:
Here's how to use `ValueIterationAgent`:

.. code-block:: python

    from deeprl.agents.value_iteration_agent import ValueIterationAgent
    from deeprl.environments import GymnasiumEnvWrapper
    import gymnasium as gym

    # Create and wrap an environment
    env = GymnasiumEnvWrapper('FrozenLake-v1')
    agent = ValueIterationAgent(env)

    # Train the agent
    agent.train()

**Method Summary**:
- `learn(self)`: Executes the value iteration algorithm to find the optimal policy.

**Attributes**:
- `value_table`: Stores the value of each state.
- `policy`: Stores the optimal policy derived from the value table.

**Details**:
The `ValueIterationAgent` class uses the value iteration algorithm to determine the optimal policy for a given environment. It iteratively updates the value function until convergence, then extracts the optimal policy from the value function.