DeterministicPolicy
===================
The `DeterministicPolicy` class implements a policy that consistently selects the same action for a given state.

.. autoclass:: deeprl.policies.DeterministicPolicy
   :members:
   :undoc-members:
   :show-inheritance:

**Example Usage**:
.. code-block:: python

    from deeprl.policies.deterministic_policy import DeterministicPolicy

    policy = DeterministicPolicy()
    action = policy.select_action([0.2, 0.5, 0.3])
