ValueIterationAgent
====================

The `ValueIterationAgent` class implements the value iteration algorithm, a dynamic programming approach used to find the optimal policy for Markov Decision Processes (MDPs) with finite state and action spaces.

.. autoclass:: deeprl.agents.value_iteration_agent.ValueIterationAgent
   :members:
   :undoc-members:
   :show-inheritance:

**Algorithm Overview**:

Value iteration is an iterative algorithm that focuses on directly optimizing the value function, :math:`V(s)`, rather than explicitly alternating between policy evaluation and improvement steps. The algorithm converges to the optimal value function :math:`V^*(s)` and then derives an optimal policy from it. Here’s a breakdown of the process:

1. **Value Function Update**:
   
   - At each iteration, the algorithm updates the value of each state :math:`V(s)` by selecting the maximum expected return across all possible actions. This is done according to the Bellman optimality equation:

     .. math::

        V(s) = \max_a \sum_{s'} P(s'|s, a) \left[ R(s, a, s') + \gamma V(s') \right]

   - Where:
     
     - :math:`P(s'|s, a)` is the probability of transitioning from state :math:`s` to :math:`s'` given action :math:`a`.
     
     - :math:`R(s, a, s')` is the reward received after transitioning from :math:`s` to :math:`s'`.
     
     - :math:`\gamma` is the discount factor that balances the importance of immediate versus future rewards.

2. **Deriving the Optimal Policy**:
   
   - Once the value function :math:`V(s)` converges to the optimal value function :math:`V^*(s)`, an optimal policy :math:`\pi^*` is derived. For each state :math:`s`, the optimal action is chosen as the action that maximizes the expected return:

     .. math::

        \pi^*(s) = \arg\max_a \sum_{s'} P(s'|s, a) \left[ R(s, a, s') + \gamma V^*(s') \right]

   - This policy is deterministic and selects the action that maximizes the value of the state according to the converged value function.

The algorithm iteratively updates the value function for each state until the values converge, meaning additional iterations do not significantly change :math:`V(s)`.

**Example Usage**:

Here's how to use `ValueIterationAgent`:

.. code-block:: python

    from deeprl.environments import GymnasiumEnvWrapper
    from deeprl.agents import ValueIterationAgent

    def main():
       # Configure the FrozenLake environment with a slippery surface using GymnasiumEnvWrapper from DeepRL
       env = GymnasiumEnvWrapper('FrozenLake-v1', render_mode='human')
       agent = ValueIterationAgent(env)
       agent.learn()

       # Unpack the initial state and reset the environment
       agent.interact(num_episodes=1, render=True)

    if __name__ == '__main__':
       main()


**Method Summary**:

- `learn(self)`: Executes the value iteration algorithm to find the optimal value function and derive the corresponding optimal policy.

**Details**:

- **Value Function Update**: In each iteration, the algorithm updates the value of each state by taking the maximum expected reward over all actions, iterating until the value function converges.

- **Policy Derivation**: After the value function has converged, the optimal policy is derived by selecting actions that maximize the expected return for each state.

- **Policy**: This agent produces a deterministic policy by default, meaning it selects actions based on the maximized value of each state. Users can implement custom policies by modifying the policy derivation process.

**Convergence**:

- Value iteration is guaranteed to converge to the optimal policy when applied to MDPs with discrete and finite state and action spaces.

- This method can be computationally intensive for large state spaces, as each iteration updates the value of every state in the environment.

**Use Cases and Limitations**:

- **Use Cases**: Value iteration is effective for small to medium-sized MDPs with discrete states and actions, where the computational cost is manageable.

- **Limitations**: For environments with very large state spaces, value iteration may be less practical due to its computational requirements. In such cases, approximate methods or other RL algorithms may be preferable.

**See Also**: 

- `deeprl.agents.PolicyIterationAgent` for a related algorithm that alternates explicitly between policy evaluation and policy improvement.
