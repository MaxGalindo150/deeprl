GymnasiumEnvWrapper
===================
The `GymnasiumEnvWrapper` class is designed to integrate Gymnasium environments with DeepRL, allowing seamless use of pre-existing environments within the library.

.. autoclass:: deeprl.environments.GymnasiumEnvWrapper
   :members:
   :undoc-members:
   :show-inheritance:

**Attributes**:
- `env`: The Gymnasium environment instance that is being wrapped.
- `reset`: Method that resets the Gymnasium environment and returns the initial state.
- `step`: Method that takes an action and returns the next state, reward, done flag, and info from the Gymnasium environment.

**Example Usage**:
Here's how to use `GymnasiumEnvWrapper` to wrap a Gymnasium environment:

.. code-block:: python

    from deeprl.environments import GymnasiumEnvWrapper
    import gymnasium as gym

    # Create and wrap a Gymnasium environment
    gym_env = gym.make('CartPole-v1')
    env = GymnasiumEnvWrapper(gym_env)

    # Use the wrapped environment
    state = env.reset()
    next_state, reward, done, info = env.step(action=1)

**Details**:
- The `GymnasiumEnvWrapper` allows users to leverage existing Gymnasium environments while maintaining compatibility with DeepRL's algorithms.
- It provides a straightforward interface that mirrors the `BaseEnvironment` class, making it easy to integrate custom logic if needed.

**Method Summary**:
- `reset(self)`: Resets the environment and returns the initial state.
- `step(self, action)`: Executes an action and returns the next state, reward, done flag, and additional information.

**See Also**:
- :class:`~deeprl.environments.BaseEnvironment` for creating custom environments from scratch.
