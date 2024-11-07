BasePolicy
==========

The `BasePolicy` class is an abstract base class for all policies in DeepRL. It defines the structure and common methods that specific policies should implement.

.. autoclass:: deeprl.policies.BasePolicy
   :members:
   :undoc-members:
   :show-inheritance:

**Attributes**:

- `select_action`: Abstract method that must be overridden by subclasses to implement action selection logic.

**Usage**:

`BasePolicy` is not meant to be used directly but should be inherited by other policy classes that implement specific action selection strategies. Here's an example of how to create a custom policy by inheriting from `BasePolicy`:

.. code-block:: python

    from deeprl.policies import BasePolicy

    class CustomPolicy(BasePolicy):
        def __init__(self):
            super().__init__()

        def select_action(self, action_values):
            # Implement custom logic for selecting an action
            return max(range(len(action_values)), key=lambda i: action_values[i])

    # Example usage of the custom policy
    policy = CustomPolicy()
    action_values = [0.1, 0.4, 0.2]
    action = policy.select_action(action_values)
    print(f"Selected action: {action}")

**Details**:

- `BasePolicy` provides a blueprint for building custom policies, ensuring consistency across different policy implementations in DeepRL.

- The method `select_action(self, action_values)` must be implemented by any subclass. It takes a list or array of action values and returns the chosen action.

**Best Practices**:

- When designing a new policy, inherit from `BasePolicy` to maintain consistency and make your policy compatible with the rest of the DeepRL library.

- Document the specific parameters and behavior of your custom `select_action` method in any subclass.

**Method Summary**:

- `select_action(self, action_values)`: Abstract method that needs to be implemented by subclasses to define how actions are selected based on input action values.

**See Also**:

- :class:`~deeprl.policies.EpsilonGreedyPolicy` for an example of a concrete policy that extends `BasePolicy`.

- :class:`~deeprl.policies.SoftmaxPolicy` for another example of a policy that implements a different strategy.
