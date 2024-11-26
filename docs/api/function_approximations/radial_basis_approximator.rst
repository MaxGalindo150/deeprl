############################
RBFBasisApproximator
############################

The ``RBFBasisApproximator`` class implements a radial basis function (RBF) approximator using scikit-learn's `RBFSampler`. This approximator transforms states into a higher-dimensional feature space, making it suitable for approximating non-linear relationships in reinforcement learning.

************************
How It Works
************************

The ``RBFBasisApproximator`` maps input states into a feature space defined by radial basis functions (RBFs). The transformation is controlled by parameters like `gamma` (the kernel width) and `n_components` (the number of basis functions). Each feature represents the similarity of the input state to a set of randomly sampled points in the feature space.

Key functionalities include:

- **Feature Computation**: Generates RBF features from states.

- **Prediction**: Computes Q-values for a given state and optionally for a specific action.

- **Update**: Adjusts the weights for the features using gradient descent to minimize the error between predicted and target values.

This approximator is particularly effective for environments with continuous state spaces where linear approximators may fail to capture complex dynamics.

***********************
Example
***********************

.. code-block:: python

    from deeprl.environments import GymnasiumEnvWrapper
    from deeprl.agents.q_learning_agent import QLearningAgent
    from deeprl.function_approximations import RBFBasisApproximator

    def main():
        
        # Initialize the environment and approximator
        env = GymnasiumEnvWrapper('CartPole-v1')
        approximator = RBFBasisApproximator(gamma=0.1, n_components=100, env=env)
            
        agent = QLearningAgent(
            env=env,
            learning_rate=0.1,
            discount_factor=0.99,
            is_continuous=True,
            approximator=approximator,
            verbose=True
        )
        
        # Train the agent
        agent.learn(episodes=10000, max_steps=10000, save_train_graph=True)
        
        # Evaluate the agent
        rewards = agent.interact(episodes=10, render=True, save_test_graph=True)

    if __name__ == '__main__':
        main()


.. note::
    Be sure to use the appropriate values for ``gamma`` and ``n_components`` based on the problem domain and the complexity of the state space. These parameters can significantly affect the performance of the approximator.

***********************
Parameters
***********************

.. autoclass:: deeprl.function_approximations.radial_basis_approximator.RBFBasisApproximator
   :members:
   :inherited-members:

***********************
See Also
***********************

- :class:`~deeprl.function_approximations.base_approximator.BaseApproximator` for the base class that `RBFBasisApproximator` inherits from.
- :class:`~deeprl.function_approximations.polynomial_basis_approximator.PolynomialBasisApproximator` for a basis function approximator using polynomial features.
- :class:`~deeprl.function_approximations.neural_network_approximator.NeuralNetworkApproximator` for a non-linear approximation approach using neural networks.

***********************
References
***********************

1. Rahimi, A., & Recht, B. (2008). *Random Features for Large-Scale Kernel Machines*. In NIPS.
2. Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction (2nd ed.)*. MIT Press. Available at: http://incompleteideas.net/book/the-book-2nd.html
3. Rasmussen, C. E., & Williams, C. K. I. (2006). *Gaussian Processes for Machine Learning*. MIT Press.
