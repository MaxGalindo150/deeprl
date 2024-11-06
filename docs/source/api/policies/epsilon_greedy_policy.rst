EpsilonGreedyPolicy
===================
The `EpsilonGreedyPolicy` class implements an exploration strategy where actions are selected randomly with a probability epsilon, promoting exploration, while the best-known action is taken with a probability of 1 - epsilon.

.. autoclass:: deeprl.policies.EpsilonGreedyPolicy
   :members:
   :undoc-members:
   :show-inheritance:

**Attributes:**
- `epsilon` (float): The exploration rate, a value between 0 and 1, indicating the probability of taking a random action.
- `select_action`: Method that selects an action based on the current policy logic.

**Example Usage**:
Here's a simple example of how to create and use an `EpsilonGreedyPolicy`:

.. code-block:: python

    from deeprl.policies.epsilon_greedy_policy import EpsilonGreedyPolicy

    # Initialize the policy with an epsilon value of 0.1 (10% exploration)
    policy = EpsilonGreedyPolicy(epsilon=0.1)

    # Simulate selecting an action from a set of action values
    action_values = [0.2, 0.5, 0.3]  # Example action values for illustration
    action = policy.select_action(action_values)

    print(f"Selected action: {action}")

**Details**:
The `EpsilonGreedyPolicy` class balances exploration and exploitation by introducing randomness in the action selection process. The `epsilon` parameter controls this randomness:
- A high `epsilon` value (e.g., 0.9) encourages more exploration.
- A low `epsilon` value (e.g., 0.1) focuses more on exploiting known actions that yield better results.

**Best Practices**:
- Adjust `epsilon` dynamically (e.g., with epsilon decay) during training to start with more exploration and gradually shift towards exploitation as the model learns.
- Use `epsilon` values that suit your problem's exploration-exploitation trade-off.

**Method Summary**:
- `select_action(self, action_values)`: Selects an action using the epsilon-greedy strategy.

**See Also**:
- :class:`~deeprl.policies.SoftmaxPolicy` for an alternative policy that uses softmax for action selection.
