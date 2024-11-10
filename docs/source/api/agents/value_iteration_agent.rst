######################
ValueIterationAgent
######################

The `ValueIterationAgent` class implements the value iteration algorithm, a dynamic programming approach used to find the optimal policy for Markov Decision Processes (MDPs) with finite state and action spaces.

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



***********************
Notes
***********************

**References**:

1. Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction (2nd ed.)*. MIT Press. Available at: http://incompleteideas.net/book/the-book-2nd.html

2. Bellman, R. E. (1957). *Dynamic Programming*. Princeton University Press.

3. Puterman, M. L. (1994). *Markov Decision Processes: Discrete Stochastic Dynamic Programming*. Wiley.

4. Bertsekas, D. P. (2007). *Dynamic Programming and Optimal Control (Vol. 1, 3rd ed.)*. Athena Scientific.


***********************
Can I use?
***********************

.. list-table::
   :header-rows: 1

   * - Space
     - Action
     - Observation
   * - Discrete
     - ✅
     - ✅
   * - Box
     - ❌
     - ❌
   * - MultiDiscrete
     - ✅
     - ✅
   * - MultiBinary
     - ✅
     - ✅


***********************
Example
***********************

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



.. autoclass:: deeprl.agents.value_iteration_agent.ValueIterationAgent
   :members:
   :inherited-members:

***********************
Parameters
***********************

***********************
See Also 
***********************

- :class:`~deeprl.agents.policy_iteration_agent.PolicyIterationAgent` for a related algorithm that alternates explicitly between policy evaluation and policy improvement.
- :class:`~deeprl.agents.q_learning_agent.QLearningAgent` for a model-free reinforcement learning algorithm that learns the optimal policy through Q-value iteration.
