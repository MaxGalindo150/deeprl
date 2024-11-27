#####################
DeepRL Documentation
#####################

`DeepRL <https://github.com/MaxGalindo150/deeprl>`_ is a Python library for basic reinforcement learning algorithms and deep reinforcement learning algorithms. It is built on top of PyTorch and provides a simple and easy-to-use interface for training and evaluating reinforcement learning agents. This library is based on the `SB3 <https://github.com/DLR-RM/stable-baselines3>`_ library and provides a similar API for training and evaluating agents.

Main Features
-------------

- Simple and easy-to-use interface for training and evaluating reinforcement learning agents.
- Built on top of PyTorch.
- PEP8 compliant.
- Documented functions and classes
- Tests, high test coverage
- Clean code
- Tensorboard support
.. change the link to the benchmarks
- The performance of each algorithm is tested (see `benchmarks <https://github.com>`)

.. toctree::
    :maxdepth: 2 
    :caption: User Guide

    guide/install
    guide/quickstart
    guide/rl_tips
    guide/rl
    guide/algorithms
    guide/examples
    guide/vec_envs
    guide/custom_policy
    guide/custom_env
    guide/callbacks
    guide/tensorboard
    guide/integration
    guide/deeprl_contributing
    guide/deeprl_faq
    guide/changelog
..    guide/deeprl_zoo
    guide/deeprl_changelog

.. toctree::
    :maxdepth: 1
    :caption: Classic RL Algorithms

    modules/q_learning

.. toctree::
    :maxdepth: 1
    :caption: Deep RL Algorithms

    modules/dqn

.. toctree::
  :maxdepth: 1
  :caption: Common

  common/atari_wrappers
  common/env_util
  common/envs
  common/distributions
  common/evaluation
  common/env_checker
  common/monitor
  common/logger
  common/noise
  common/utils



