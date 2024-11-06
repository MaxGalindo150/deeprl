SoftmaxPolicy
=============
The `SoftmaxPolicy` class implements a policy that selects actions based on the softmax distribution.

.. autoclass:: deeprl.policies.SoftmaxPolicy
   :members:
   :undoc-members:
   :show-inheritance:

**Example Usage**:
.. code-block:: python

    from deeprl.policies.softmax_policy import SoftmaxPolicy

    policy = SoftmaxPolicy(temperature=1.0)
    action = policy.select_action([0.2, 0.5, 0.3])
