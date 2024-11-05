Introduction to DeepRL
======================

DeepRL is a reinforcement learning library designed to simplify the development, experimentation, and deployment of reinforcement learning algorithms. With a modular design, DeepRL is ideal for both researchers and developers looking to explore and build RL solutions quickly and efficiently.

**Key Features:**

- **Modular and extensible**: Adaptable to your needs, whether for quick prototyping or complex research projects.

- **PyTorch integration**: Built on PyTorch to ensure flexibility and high performance.

- **Gymnasium integration**: Seamlessly integrates with OpenAI Gymnasium environments for easy experimentation.

- **Practical examples**: Comes with ready-to-use examples to get you started quickly.

**Why use DeepRL?**
DeepRL combines the simplicity of an intuitive design with the power of best practices in reinforcement learning. Unlike other libraries, it offers a clear and extensible architecture that makes it easy to customize algorithms and integrate with other projects.

**General Architecture:**
DeepRL is structured into distinct modules to promote modularity and ease of use. Below is an overview of the main components:

- `agents`: Contains the base agent class (`base_agent.py`) and specific agent implementations such as `dqn.py`, `ppo.py`, and `sarsa.py`, and others. This module provides a flexible framework for developing and managing agents.

- `environments`: Includes `base_environment.py` for defining custom environment classes and `gymnasium_env_wrapper.py` for integrating with Gymnasium-based environments. This module helps in managing different types of training and testing environments seamlessly.
  
- `policies`: Defines various policy strategies, including `base_policy.py` for the policy interface and specialized implementations like `deterministic_policy.py`, `epsilon_greedy_policy.py`, and `softmax_policy.py`. This module allows for flexible policy management.


**Directory Overview:**
The project structure ensures that related classes and implementations are organized within their respective directories, making it intuitive for developers to locate base classes and extend functionalities.


**Quick Example:**
Here's a quick example of how to train an agent using DeepRL with the DQN algorithm:

.. code-block:: python

    from deeprl.agents import DQN
    from deeprl.environments import GymasiumEnvWrapper

    # Initialize the environment
    env = GymEnvWrapper('CartPole-v1')

    # Initialize the DQN agent
    agent = DQN(env)

    # Train the agent
    agent.learn(episodes=100)

    # Evaluate the agent
    agent.evaluate()

This example demonstrates how easy it is to get started with DeepRL by training an agent in a classic control environment.

**Next Steps:**
To learn more about how to install DeepRL and set up your development environment, check out the **Installation** section.
