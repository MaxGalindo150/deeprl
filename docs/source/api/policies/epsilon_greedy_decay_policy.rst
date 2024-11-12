###############################
EpsilonGreedyDecayPolicy
###############################

The `EpsilonGreedyDecayPolicy` class extends the epsilon-greedy exploration strategy by introducing a decay mechanism for epsilon. This allows the policy to gradually shift from exploration to exploitation over time, balancing the trade-off dynamically as learning progresses.

The `EpsilonGreedyDecayPolicy` introduces two additional parameters, `decay_rate` and `min_epsilon`, to control the rate of decay and the lower bound for epsilon, respectively. This ensures that the agent doesn't stop exploring entirely.

************************
How It Works
************************

1. **Exploration**: With a probability of :math:`\epsilon`, the agent selects a random action to discover new strategies or areas of the state space.

2. **Exploitation**: With a probability of :math:`1 - \epsilon`, the agent selects the action that maximizes the current Q-value.

3. **Decay Mechanism**: After each step, epsilon is decayed by multiplying it with the decay rate, ensuring that over time, the agent shifts towards exploitation. The epsilon value never goes below the specified `min_epsilon`.

************************
Example
************************

.. code-block:: python

    from deeprl.policies.epsilon_greedy_decay_policy import EpsilonGreedyDecayPolicy
    import torch

    # Initialize the policy with epsilon=0.5, decay_rate=0.95, and min_epsilon=0.1
    policy = EpsilonGreedyDecayPolicy(epsilon=0.5, decay_rate=0.95, min_epsilon=0.1)

    # Simulate selecting an action from a set of Q-values
    q_values = torch.tensor([0.3, 0.7, 0.2])  # Example Q-values
    action = policy.select_action(q_values)

    print(f"Selected action: {action}")
    print(f"Current epsilon: {policy.epsilon}")

    # Update epsilon with decay
    policy.update()
    print(f"Updated epsilon: {policy.epsilon}")

************************
Parameters
************************

.. autoclass:: deeprl.policies.epsilon_greedy_decay_policy.EpsilonGreedyDecayPolicy
   :members:
   :inherited-members:

************************
See Also
************************

- :class:`~deeprl.policies.base_policy.BasePolicy` for the base class that `EpsilonGreedyDecayPolicy` inherits from.
- :class:`~deeprl.policies.epsilon_greedy_policy.EpsilonGreedyPolicy` for the standard epsilon-greedy policy without decay.
- :class:`~deeprl.policies.softmax_policy.SoftmaxPolicy` for a policy that selects actions based on the softmax strategy.
- :class:`~deeprl.policies.deterministic_policy.DeterministicPolicy` for a policy that selects deterministic actions for each state.
