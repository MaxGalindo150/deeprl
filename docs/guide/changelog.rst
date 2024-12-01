Changelog
=========

All notable changes to this project will be documented in this file. This project adheres to `Semantic Versioning <https://semver.org/>`_.

Unreleased
-----------

**Added**

- **Policy module**:
  - ``SoftmaxPolicy`` for stochastic action selection based on action preferences.
- **Deep RL algorithms**:
  - ``DQN`` (Deep Q-Network) for continuous and high-dimensional state spaces.
  - ``PPO`` (Proximal Policy Optimization) for policy gradient optimization.
  - ``A3C`` (Asynchronous Advantage Actor-Critic) for distributed learning.
  - ``SAC`` (Soft Actor-Critic) for maximum entropy reinforcement learning.
  - ``DDPG`` (Deep Deterministic Policy Gradient) for continuous action spaces.
- **Neural networks module**:
  - Utilities for creating custom neural networks using PyTorch.
  - Predefined architectures for DQN, PPO, and SAC agents.
- **Examples**:
  - Deep RL examples showcasing the training of ``DQN`` in ``CartPole``.
  - PPO implementation with custom reward shaping in continuous control environments like ``LunarLanderContinuous``.

**Changed**

- Optimized ``EpsilonGreedyDecayPolicy`` for faster epsilon updates during training.
- Improved integration between function approximators and agents for seamless configuration.

**Fixed**

- Bug in reward shaping logic for MountainCar where shaped rewards could incorrectly accumulate negative values.
- Fixed compatibility issues in ``RBFBasisApproximator`` with environments that use high-dimensional state spaces.

0.1.0 - 2024-11-09
-------------------

**Added**

- **Core agent implementations**:
  
  - ``QLearningAgent`` supporting both discrete and continuous state spaces using function approximators.
  - ``PolicyIterationAgent`` and ``ValueIterationAgent`` for solving MDPs.
- **Function approximation module**:
  
  - ``PolynomialBasisApproximator`` for polynomial feature transformations.
  - ``RBFBasisApproximator`` for radial basis function (RBF) feature transformations.
- **Reward shaping module**:
  - Base class ``BaseRewardShaping`` and custom implementations:

    - ``DefaultReward`` for step penalties.
    - ``DistanceBasedShaping`` for shaping rewards based on distance to a goal.
    - ``MountainCarRewardShaping`` for guiding agents in the MountainCar environment.
    - ``PotentialBasedShaping`` for reward shaping using potential functions.
- **Policy module**:

  - ``EpsilonGreedyPolicy`` for exploration-exploitation tradeoff.
  - ``EpsilonGreedyDecayPolicy`` with support for decaying epsilon over time.
- **Environment wrapper**:
  
  - ``GymnasiumEnvWrapper`` to seamlessly integrate Gymnasium environments.
- **Visualization tools**:
  
  - ``ProgressBoard`` for monitoring and saving training progress.
- **Utilities**:
  
  - Progress printing for agents with detailed statistics.
- **Example scripts**:
  
  - ``run_q_learning_agent_w_approximation_function.py`` demonstrating the use of ``QLearningAgent`` with function approximators and reward shaping.
  
  - Integration of the ``MountainCar`` environment with custom reward shaping.
- **Comprehensive unit tests**:
  
  - Function approximators (``PolynomialBasisApproximator``, ``RBFBasisApproximator``).
  - Reward shaping implementations.
  - Core agent behaviors.

**Changed**

- Improved rendering logic in ``QLearningAgent`` to reuse the same environment instance for rendering.
- Enhanced modularity in ``QLearningAgent`` to accept custom reward shaping functions for better flexibility.

**Fixed**

- Corrected dimensionality handling in function approximators to ensure compatibility with various state spaces.
- Addressed bugs in ``GymnasiumEnvWrapper`` for accurate environment resets and rendering configurations.
