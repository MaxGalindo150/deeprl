# deeprlearn

[![PyPI version](https://badge.fury.io/py/deeprl.svg)](https://badge.fury.io/py/deeprlearn)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

**deeprlearn** is a modular reinforcement learning library built on PyTorch. Designed for researchers and developers, it provides a robust framework to experiment with, implement, and optimize RL algorithms for small to medium-scale environments. Check out the [documentation](http://deeprl.sytes.net/) for more details.

---

## Key Features

- **Dynamic Programming Agents**:
  - Implementation of **Value Iteration** and **Policy Iteration** algorithms.
- **Function Approximations**:
  - Support for linear and non-linear function approximation using techniques like **Radial Basis Functions (RBF)**, **Polynomial Features**, and **Neural Networks**.
- **Reward Shaping**:
  - Includes strategies like **Distance-Based Shaping**, **Potential-Based Shaping**, and **Step Penalty Shaping** to improve learning in sparse reward environments.
- **Seamless Integration**:
  - Compatible with **Gymnasium environments**, simplifying the setup and testing of RL agents.
- **Progress Monitoring**:
  - Verbose mode for tracking rewards, steps, and exploration rates during training.
- **Model Persistence**:
  - Save and load models easily for reproducibility and testing.

---

## Installation

Install **deeprl** directly from PyPI:

```bash
pip install deeprlearn
```

### Requirements

- Python 3.9 or higher
- Dependencies:
  - NumPy
  - PyTorch
  - Gymnasium
  - Scikit-learn

---

## Quick Start

Here’s how to train a **Q-Learning** agent on the `MountainCar` environment:

```python
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
```

---

## Features Overview

### 1. Dynamic Programming Agents
- **ValueIterationAgent**: Uses the Value Iteration algorithm to compute the optimal policy.
- **PolicyIterationAgent**: Implements the Policy Iteration algorithm for policy optimization.

### 2. Function Approximations
Support for approximating value functions or policies using:
- **Linear Approximators**: Efficient for linearly separable problems.
- **Radial Basis Function (RBF) Approximators**: Captures non-linear patterns in continuous spaces.
- **Polynomial Approximators**: Expands features for higher-dimensional representation.
- **Neural Network Approximators**: For complex, non-linear function approximation.

### 3. Reward Shaping
Enhance learning in sparse or uninformative environments with:
- **Distance-Based Shaping**: Rewards progress toward a specific goal.
- **Potential-Based Shaping**: Ensures policy invariance while guiding exploration.
- **Step Penalty Shaping**: Penalizes excessive steps to encourage efficiency.

### 4. Integration with Gymnasium
Easily integrate with a wide range of Gymnasium environments, supporting both discrete and continuous action/state spaces.

### 5. Saving and Loading Agents
Persist agent parameters for reproducibility or testing:

```python
# Save the agent's parameters
agent.save('value_iteration_agent.pkl')

# Load the agent's parameters
agent.load('value_iteration_agent.pkl')
```

---

## Contribution Guidelines

Contributions are welcome! To contribute, follow these steps:

1. **Fork the repository**:
   ```bash
   git clone https://github.com/MaxGalindo150/deeprl.git
   ```
2. **Create a new branch**:
   ```bash
   git checkout -b feature/new-feature
   ```
3. **Make your changes and commit**:
   ```bash
   git commit -am 'Add new feature'
   ```
4. **Push your branch**:
   ```bash
   git push origin feature/new-feature
   ```
5. **Open a pull request** on the main repository.

---

## License

This project is licensed under the **MIT License**. See the [LICENSE](https://github.com/MaxGalindo150/deeprl/blob/main/LICENSE) file for more details.

---

## Contact

For inquiries or collaboration, feel free to reach out:

- **Author**: Maximiliano Galindo  
- **Email**: [maximilianogalindo7@gmail.com](mailto:maximilianogalindo7@gmail.com)


