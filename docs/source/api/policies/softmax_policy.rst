SoftmaxPolicy
=============

The `SoftmaxPolicy` class implements a stochastic policy that selects actions based on a softmax distribution over action preferences. This policy introduces randomness in action selection, making it suitable for exploration in reinforcement learning algorithms.

.. warning::  
   The `SoftmaxPolicy` class is under development and may exhibit instability in certain scenarios. Future releases will address these issues and enhance its performance and flexibility. Use with caution in critical applications.

************************
How It Works
************************

The ``SoftmaxPolicy`` converts a list of action preferences (e.g., Q-values or logits) into probabilities using the softmax function:

.. math::

    P(a_i) = \frac{e^{h_i / T}}{\sum_{j} e^{h_j / T}}

Where:

- :math:`P(a_i)` is the probability of selecting action :math:`a_i`.
- :math:`h_i` is the preference for action :math:`a_i`.
- :math:`T` is the temperature parameter that controls the randomness of action selection:
  - Higher :math:`T` values produce more uniform probabilities, increasing exploration.
  - Lower :math:`T` values make the policy more deterministic, favoring actions with higher preferences.

The `SoftmaxPolicy` is particularly useful in reinforcement learning settings where balancing exploration and exploitation is crucial.

***********************
Example Usage
***********************

Here’s an example of how to use the `SoftmaxPolicy` class:

.. code-block:: python

    from deeprl.policies.softmax_policy import SoftmaxPolicy

    # Initialize the softmax policy with a temperature of 1.0
    policy = SoftmaxPolicy(temperature=1.0)

    # Define action preferences (e.g., Q-values or logits)
    action_preferences = [0.2, 0.5, 0.3]

    # Select an action based on the softmax distribution
    action = policy.select_action(action_preferences)

    print(f"Selected action: {action}")

***********************
Parameters
***********************

.. autoclass:: deeprl.policies.softmax_policy.SoftmaxPolicy
   :members:
   :undoc-members:
   :show-inheritance:

************************
See Also
************************

- :class:`~deeprl.policies.BasePolicy` for the base class that `SoftmaxPolicy` inherits from.
- :class:`~deeprl.policies.EpsilonGreedyPolicy` for a policy that uses an epsilon-greedy exploration strategy.
- :class:`~deeprl.policies.DeterministicPolicy` for a deterministic policy that always selects the highest-valued action.
