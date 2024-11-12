Introduction to deeprl
======================

deeprl is a reinforcement learning library designed to simplify the development, experimentation, and deployment of reinforcement learning algorithms. With a modular design, deeprl is ideal for both researchers and developers looking to explore and build RL solutions quickly and efficiently.

**Key Features:**

- **Modular and extensible**: Adaptable to your needs, whether for quick prototyping or complex research projects.

- **PyTorch integration**: Built on PyTorch to ensure flexibility and high performance.

- **Gymnasium integration**: Seamlessly integrates with OpenAI Gymnasium environments for easy experimentation.

- **Practical examples**: Comes with ready-to-use examples to get you started quickly.

**Why use deeprl?**
deeprl combines the simplicity of an intuitive design with the power of best practices in reinforcement learning. Unlike other libraries, it offers a clear and extensible architecture that makes it easy to customize algorithms and integrate with other projects.

**General Architecture:**
deeprl is structured into distinct modules to promote modularity and ease of use. Below is an overview of the main components:

- ``agents``: Contains the base agent class (``base_agent.py``) and specific agent implementations such as ``dqn.py``, ``ppo.py``, and ``sarsa.py``, and others. This module provides a flexible framework for developing and managing agents.

- ``environments``: Includes ``base_environment.py`` for defining custom environment classes and ``gymnasium_env_wrapper.py`` for integrating with Gymnasium-based environments. This module helps in managing different types of training and testing environments seamlessly.
  
- ``policies``: Defines various policy strategies, including ``base_policy.py`` for the policy interface and specialized implementations like ``deterministic_policy.py``, ``epsilon_greedy_policy.py``, and ``softmax_policy.py``. This module allows for flexible policy management.


**Directory Overview:**
The project structure ensures that related classes and implementations are organized within their respective directories, making it intuitive for developers to locate base classes and extend functionalities.


**Quick Example:**
Here's a quick example of how to train an agent using deeprl with the Q-Learning algorithm with function approximation and reward shaping:

.. code-block:: python

    from deeprl.environments import GymnasiumEnvWrapper
    from deeprl.agents.q_learning_agent import QLearningAgent
    from deeprl.function_approximations import RBFBasisApproximator
    from deeprl.reward_shaping import MountainCarRewardShaping

    def main():
        
        # Initialize the environment and approximator
        env = GymnasiumEnvWrapper('MountainCar-v0')
        approximator = RBFBasisApproximator(env=env, gamma=0.5, n_components=500)
            
        agent = QLearningAgent(
            env=env,
            learning_rate=0.1,
            discount_factor=0.99,
            is_continuous=True,
            approximator=approximator,
            reward_shaping=MountainCarRewardShaping(),
            verbose=True
        )
        
        # Train the agent
        agent.learn(episodes=10000, max_steps=10000, save_train_graph=True)
        
        # Evaluate the agent
        rewards = agent.interact(episodes=10, render=True, save_test_graph=True)

    if __name__ == '__main__':
        main()

This example demonstrates how easy it is to get started with deeprl by training an agent in a classic control environment.

**Next Steps:**
To learn more about how to install deeprl and set up your development environment, check out the **Installation** section.