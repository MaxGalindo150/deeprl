Here's an English version of your `README.md` for **DeepRL**:

```markdown
# DeepRL

[![PyPI version](https://badge.fury.io/py/deeprl.svg)](https://badge.fury.io/py/deeprl)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

**DeepRL** is a deep reinforcement learning library based on PyTorch. It is designed for advanced researchers and developers looking for a flexible and extensible framework to implement, test, and optimize reinforcement learning algorithms.

## Features

- Implementation of dynamic programming algorithms like Value Iteration and Policy Iteration.
- Seamless integration with Gymnasium environments.
- Support for saving and loading trained models.
- Modular and easy-to-extend interface.

## Installation

You can install **DeepRL** from PyPI with:

```bash
pip install deeprl
```

### Prerequisites

- Python 3.9 or higher
- Dependencies: NumPy, Pytorch, Gymnasium, among others.

## Getting Started

Here's a quick example of how to use **DeepRL** to train a Value Iteration agent on `FrozenLake-v1`:

```python
import gymnasium
from deeprl.agents import ValueIterationAgent

# Create the environment
env = gymnasium.make('FrozenLake-v1', is_slippery=False)

# Create and train the agent
agent = ValueIterationAgent(env)
agent.learn()

# Test the agent
state, _ = env.reset()
done = False
env.render()

while not done:
    action = agent.act(state)
    state, reward, done, truncated, info = env.step(action)
    env.render()

env.close()
```

## Features Overview

### 1. Dynamic Programming Agents
- **ValueIterationAgent**: An agent that uses the Value Iteration algorithm to learn the optimal policy.
- **PolicyIterationAgent**: An agent that uses the Policy Iteration algorithm.

### 2. Integration with Gymnasium
**DeepRL** integrates smoothly with Gymnasium environments, enabling experimentation with various reinforcement learning scenarios.

## Saving and Loading Agents

You can easily save and load the agent's parameters using the `save()` and `load()` methods:

```python
# Save the agent's parameters
agent.save('value_iteration_agent.pkl')

# Load the agent's parameters
agent.load('value_iteration_agent.pkl')
```

## Contributions

Contributions are welcome! If you'd like to contribute, please follow these steps:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature/new-feature`).
3. Make your changes and commit them (`git commit -am 'Add new feature'`).
4. Push your branch (`git push origin feature/new-feature`).
5. Open a pull request.

## License

This project is licensed under the MIT License. For more details, check the [LICENSE](https://github.com/MaxGalindo150/DeepRL/blob/main/LICENSE) file.

## Contact

Author: Maximiliano Galindo  
Email: maximilianogalindo7@gmail.com

---

Feel free to customize the content to better fit your library's needs or style.