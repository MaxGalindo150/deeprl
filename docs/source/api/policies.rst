Policies Module
===============
The `policies` module includes various policy implementations used in reinforcement learning.

.. automodule:: deeprl.policies
   :members:
   :undoc-members:
   :show-inheritance:

**Key Classes:**

- :class:`BasePolicy`: The foundational class for policy implementations.
- :class:`DeterministicPolicy`: A policy that selects the same action for a given state.
- :class:`EpsilonGreedyPolicy`: A policy that selects actions randomly with a probability epsilon.
- :class:`SoftmaxPolicy`: A policy that uses the softmax function to choose actions.

**Example Usage**:
.. code-block:: python

    from deeprl.policies.epsilon_greedy_policy import EpsilonGreedyPolicy

    policy = EpsilonGreedyPolicy(epsilon=0.1)
    action = policy.select_action([0.2, 0.5, 0.3])