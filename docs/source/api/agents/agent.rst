Agent
=========
The `Agent` class is the foundational class for all agent implementations in DeepRL. It provides common interfaces and methods that other agent classes extend.

.. autoclass:: deeprl.agents.base_agent.Agent
   :members:
   :undoc-members:
   :show-inheritance:

**Example Usage**:
Here's how to create a custom agent that extends `Agent`:

.. code-block:: python

    from deeprl.agents.base_agent import Agent

    class CustomAgent(Agent):
        def __init__(self, env):
            super().__init__(env)
            # Initialize custom parameters

        def learn(self):
            # Implement custom training logic
            pass

**Attributes and Methods**:
- `learn(self)`: Abstract method that must be implemented by subclasses for training logic.
- `evaluate(self)`: Optional method for evaluating the agent's performance.

**Details**:
The `Agent` class provides a structure that ensures consistency across different agent implementations in DeepRL.
