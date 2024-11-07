####################
GymnasiumEnvWrapper
####################

The `GymnasiumEnvWrapper` class is designed to integrate Gymnasium environments with DeepRL, allowing seamless use of pre-existing environments within the library.

**Details**:

- The `GymnasiumEnvWrapper` allows users to leverage existing Gymnasium environments while maintaining compatibility with DeepRL's algorithms.

- It provides a straightforward interface that mirrors the `BaseEnvironment` class, making it easy to integrate custom logic if needed.


************************************
Example
************************************

.. code-block:: python

    from deeprl.environments import GymnasiumEnvWrapper
    import gymnasium as gym

    # Create and wrap a Gymnasium environment
    gym_env = gym.make('CartPole-v1')
    env = GymnasiumEnvWrapper(gym_env)

    # Use the wrapped environment
    state = env.reset()
    next_state, reward, done, info = env.step(action=1)


************************************
Parameters
************************************

.. autoclass:: deeprl.environments.GymnasiumEnvWrapper
   :members:
   :inherited-members:


************************************
See Also
************************************

- :class:`~deeprl.environments.BaseEnvironment` for creating custom environments from scratch.
