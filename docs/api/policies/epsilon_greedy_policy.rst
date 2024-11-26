#####################
EpsilonGreedyPolicy
#####################

The `EpsilonGreedyPolicy` class implements an exploration strategy where actions are selected randomly with a probability epsilon, promoting exploration, while the best-known action is taken with a probability of :math:`1 - \epsilon`.

************************
How It Works
************************

The `EpsilonGreedyPolicy` class balances exploration and exploitation by introducing randomness in the action selection process. The `epsilon` parameter controls this randomness:

- A high `epsilon` value (e.g., 0.9) encourages more exploration.

- A low `epsilon` value (e.g., 0.1) focuses more on exploiting known actions that yield better results.


***********************
Example
***********************

.. code-block:: python

    from deeprl.policies.epsilon_greedy_policy import EpsilonGreedyPolicy

    # Initialize the policy with an epsilon value of 0.1 (10% exploration)
    policy = EpsilonGreedyPolicy(epsilon=0.1)

    # Simulate selecting an action from a set of action values
    action_values = [0.2, 0.5, 0.3]  # Example action values for illustration
    action = policy.select_action(action_values)

    print(f"Selected action: {action}")


***********************
Parameters
***********************

.. autoclass:: deeprl.policies.epsilon_greedy_policy.EpsilonGreedyPolicy
   :members:
   :inherited-members:

***********************
See Also
***********************

- :class:`~deeprl.policies.base_policy.BasePolicy` for the base class that `EpsilonGreedyPolicy` inherits from.
- :class:`~deeprl.policies.epsilon_greedy_decay_policy.EpsilonGreedyDecayPolicy` for a policy that selects actions based on the epsilon-greedy strategy with a decaying epsilon value.
- :class:`~deeprl.policies.softmax_policy.SoftmaxPolicy` for a policy that selects actions based on the softmax strategy.
- :class:`~deeprl.policies.deterministic_policy.DeterministicPolicy` for a policy that consistently selects the same action for a given state.

