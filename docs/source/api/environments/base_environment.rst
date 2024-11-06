BaseEnvironment
===============
The `BaseEnvironment` class serves as an abstract base class for defining custom environments in DeepRL. It outlines the essential methods that must be implemented to create a fully functional environment.

.. autoclass:: deeprl.environments.BaseEnvironment
   :members:
   :undoc-members:
   :show-inheritance:

**Attributes**:
- `reset`: Abstract method that should be overridden to reset the environment and return the initial state.
- `step`: Abstract method that should be overridden to take an action and return the next state, reward, done flag, and additional info.
- `render`: Optional method for rendering the environment's current state.

**Example Usage**:
`BaseEnvironment` is intended to be inherited by other classes to create custom environments. Here's an example of how to create a custom environment:

.. code-block:: python

    from deeprl.environments import BaseEnvironment

    class CustomEnvironment(BaseEnvironment):
        def __init__(self):
            super().__init__()

        def reset(self):
            # Implement reset logic
            return initial_state

        def step(self, action):
            # Implement step logic
            next_state, reward, done, info = ...
            return next_state, reward, done, info

        def render(self):
            # Optionally implement rendering logic
            pass

    # Example usage of the custom environment
    env = CustomEnvironment()
    state = env.reset()
    next_state, reward, done, info = env.step(action=0)

**Details**:
- `BaseEnvironment` ensures that all custom environments adhere to a consistent interface, making it easier to integrate them with DeepRL algorithms.
- The `reset()` and `step()` methods are required for the environment to function properly, while `render()` is optional.

**Method Summary**:
- `reset(self)`: Resets the environment and returns the initial state.
- `step(self, action)`: Executes an action and returns the next state, reward, done flag, and additional information.
- `render(self)`: Optionally renders the current state of the environment.

**See Also**:
- :class:`~deeprl.environments.GymnasiumEnvWrapper` for an example of an environment wrapper that integrates with Gymnasium environments.
