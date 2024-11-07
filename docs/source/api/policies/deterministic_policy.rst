####################
DeterministicPolicy
####################

The `DeterministicPolicy` class implements a policy that consistently selects the same action for a given state.

***********************
Example
***********************

.. code-block:: python

    from deeprl.policies.deterministic_policy import DeterministicPolicy

    policy = DeterministicPolicy()
    action = policy.select_action([0.2, 0.5, 0.3])


***********************
Parameters
***********************

.. autoclass:: deeprl.policies.deterministic_policy.DeterministicPolicy
   :members:
   :inherited-members:

***********************
See Also
***********************

- :class:`~deeprl.policies.base_policy.BasePolicy` for the base class that `DeterministicPolicy` inherits from.
- :class:`~deeprl.policies.epsilon_greedy_policy.EpsilonGreedyPolicy` for a policy that selects actions based on the epsilon-greedy strategy.
- :class:`~deeprl.policies.softmax_policy.SoftmaxPolicy` for a policy that selects actions based on the softmax strategy.