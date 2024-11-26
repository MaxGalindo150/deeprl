###############################
PolynomialBasisApproximator
###############################

The ``PolynomialBasisApproximator`` class implements a polynomial basis function approximator using scikit-learn's ``PolynomialFeatures``. This approximator allows for feature expansion by generating polynomial combinations of the input features, making it useful for approximating non-linear value functions in reinforcement learning.

************************
How It Works
************************

The ``PolynomialBasisApproximator`` generates polynomial features of the input state using scikit-learn's ``PolynomialFeatures``. The weights for these features are maintained as a PyTorch tensor, allowing efficient updates during training.

Key functionalities include:

- **Feature Computation**: Expands input states into polynomial features.

- **Prediction**: Computes the estimated Q-values for given states using the current weights.

- **Update**: Adjusts the weights based on the target Q-values using a learning rate.

This approximator is ideal for environments with continuous state spaces where linear function approximation is insufficient to capture complex relationships.


************************
Example
************************

.. code-block:: python

    from deeprl.environments import GymnasiumEnvWrapper
    from deeprl.agents.q_learning_agent import QLearningAgent
    from deeprl.function_approximations import PolynomialBasisApproximator

    def main():
        
        # Initialize the environment and approximator
        env = GymnasiumEnvWrapper('CartPole-v1')
        approximator = PolynomialBasisApproximator(env=env, degree=2)
            
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


***********************
Parameters
***********************

.. autoclass:: deeprl.function_approximations.polynomial_basis_approximator.PolynomialBasisApproximator
   :members:
   :inherited-members:

***********************
See Also
***********************

- :class:`~deeprl.function_approximations.base_approximator.BaseApproximator` for the base class that ``PolynomialBasisApproximator`` inherits from.
- :class:`~deeprl.function_approximations.radial_basis_approximator.RBFBasisApproximator` for an approximator that uses radial basis functions for feature expansion.