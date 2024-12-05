
# deeprlearn

[![PyPI version](https://badge.fury.io/py/deeprlearn.svg)](https://badge.fury.io/py/deeprlearn)  
<!-- [![CI](https://github.com/MaxGalindo150/deeprl/workflows/CI/badge.svg)](https://github.com/MaxGalindo150/deeprl/actions/workflows/ci.yml)  
[![Documentation Status](https://readthedocs.org/projects/deeprlearn/badge/?version=latest)](https://deeprlearn.readthedocs.io/en/latest/?badge=latest)   -->
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)  
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

---

<!-- <img src="docs/static/img/logo.png" align="right" width="40%"/> -->

**deeprlearn** is a modular reinforcement learning library built on **PyTorch** and heavily inspired by the architecture of **Stable-Baselines3 (SB3)**. Initially designed for single-agent algorithms, **deeprlearn** is now expanding to support **multi-agent reinforcement learning (MARL)** and **multi-objective tasks**, enabling solutions for more complex and interactive problems.

This project is being developed by **Maximiliano Galindo** and **EigenCore**, aiming to provide an accessible and powerful tool for researchers and developers.

---

## Key Features

| **Feature**                      | **Current Status**    |
| --------------------------------- | --------------------- |
| State-of-the-art RL methods       | :heavy_check_mark:    |
| Documentation                     | :heavy_check_mark:    |
| Support for custom environments   | :heavy_check_mark:    |
| Custom policies                   | :heavy_check_mark:    |
| Common interface                  | :heavy_check_mark:    |
| Multi-objective task support      | :construction: *(In Progress)* |
| Multi-agent learning (MARL)       | :construction: *(In Progress)* |
| Gymnasium compatibility           | :heavy_check_mark:    |
| IPython/Notebook friendly         | :heavy_check_mark:    |
| TensorBoard support               | :heavy_check_mark:    |
| PEP8 code style                   | :heavy_check_mark:    |
| Custom callbacks                  | :heavy_check_mark:    |
| High test coverage                | :construction: *(Expanding)* |
| Type hints                        | :heavy_check_mark:    |

---

## Expansion to Multi-Agent and Multi-Objective Learning

**deeprlearn** is actively being expanded to include:

1. **Multi-Agent Reinforcement Learning (MARL)**:
   - Initial implementations of algorithms like **Multi-Agent PPO (MAPPO)** and **MADDPG**.
   - Support for complex interaction environments, compatible with **PettingZoo** and custom-built environments.
   - Centralized training and decentralized execution for cooperative and competitive scenarios.

2. **Multi-Objective Tasks**:
   - Policy optimization for conflicting objectives using approaches such as:
     - Objective weighting.
     - Pareto fronts for non-dominated solutions.
   - Designed for problems in ecological simulations, traffic systems, and urban planning.

---

## Installation

**Note:** **deeprlearn** requires Python 3.9 or higher.

Install directly from PyPI:
```bash
pip install deeprlearn
```

---

## Quick Start Example

Train a **Q-Learning** agent in the `MountainCar` environment:

```python
import gymnasium as gym
from deeprl import PPO
from deeprl.common.env_util import make_vec_env

# Parallel environments
vec_env = make_vec_env("CartPole-v1", n_envs=4)

model = PPO("MlpPolicy", vec_env, verbose=1)
model.learn(total_timesteps=25000)
model.save("ppo_cartpole")

del model # remove to demonstrate saving and loading

model = PPO.load("ppo_cartpole")
obs = vec_env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = vec_env.step(action)
    vec_env.render("human")
```

---

## Documentation

Detailed documentation is available online: [deeprlearn Documentation](http://deeprl.sytes.net/).

---

## Contribution Guidelines

We welcome contributions! To contribute:

1. **Fork the repository**:
   ```bash
   git clone https://github.com/MaxGalindo150/deeprl.git
   ```
2. **Create a new branch**:
   ```bash
   git checkout -b feature/new-feature
   ```
3. **Make changes and commit**:
   ```bash
   git commit -am 'Add new feature'
   ```
4. **Push your branch**:
   ```bash
   git push origin feature/new-feature
   ```
5. **Open a pull request** on the main repository.

---

## Contact

For inquiries or collaboration, feel free to reach out:

- **Author**: Maximiliano Galindo  
- **Email**: [maximilianogalindo7@gmail.com](mailto:maximilianogalindo7@gmail.com)  
- **Organization**: [EigenCore](https://eigen.core)  

