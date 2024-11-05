Agents Module
=============
The `agents` module contains classes for reinforcement learning agents, including base classes and specific implementations.

.. automodule:: deeprl.agents
   :members:
   :undoc-members:
   :show-inheritance:

**Key Classes:**

- **`BaseAgent`**: The foundational class for all agent implementations.
  
  **Example Usage**:
  .. code-block:: python

      from deeprl.agents.base_agent import BaseAgent

      class CustomAgent(BaseAgent):
          def __init__(self, env):
              super().__init__(env)
              # Initialize custom parameters

          def learn(self):
              # Implement learn logic
              pass

- **`PolicyIterationAgent`**: Implements the policy iteration algorithm.
- **`ValueIterationAgent`**: Implements the value iteration algorithm.
