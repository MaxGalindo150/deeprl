#######################
BaseApproximator
#######################

The ``BaseApproximator`` class is an abstract base class for all function approximators in DeepRL. It defines the structure and common methods that specific approximators should implement.

.. autoclass:: deeprl.function_approximations.BaseApproximator
   :members:
   :undoc-members:
   :show-inheritance:

***********************
Example
***********************

``BaseApproximator`` is not meant to be used directly but should be inherited by other approximator classes that implement specific function approximation techniques. Here's an example of how to create a custom approximator by inheriting from `BaseApproximator`:

.. code-block:: python

    from deeprl.function_approximations import BaseApproximator

    class CustomApproximator(BaseApproximator):
        def __init__(self):
            super().__init__()

        def predict(self, state):
            # Implement custom logic for approximating the value of a state
            return 0.0

    # Example usage of the custom approximator
    approximator = CustomApproximator()
    state = [0.1, 0.2, 0.3]
    value = approximator.predict(state)
    print(f"Approximated value: {value}")

***********************
Best Practices
***********************

- When designing a new approximator, inherit from ``BaseApproximator`` to maintain consistency and make your approximator compatible with the rest of the DeepRL library.

- Document the specific parameters and behavior of your custom ``predict`` method in any subclass.

***********************
See Also
***********************
- :class:`~deeprl.function_approximations.PolynomialBasisApproximator` for an example of a concrete approximator that extends `BaseApproximator`.

- :class:`~deeprl.function_approximations.RadialBasisApproximator` for another example of an approximator that implements a different approximation technique.

