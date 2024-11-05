Environments Module
===================
The `environments` module provides classes for creating and managing simulation environments.

.. automodule:: deeprl.environments
   :members:
   :undoc-members:
   :show-inheritance:

**Key Classes:**

- **`BaseEnvironment`**: Base class for creating custom environments.
- **`GymnasiumEnvWrapper`**: Wrapper for integrating Gymnasium environments.

**Example Usage**:
.. code-block:: python

    from deeprl.environments.gymnasium_env_wrapper import GymnasiumEnvWrapper

    env = GymnasiumEnvWrapper('CartPole-v1')
    observation = env.reset()
