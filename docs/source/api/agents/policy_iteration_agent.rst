
####################################
PolicyIterationAgent
####################################

The `PolicyIterationAgent` class implements the policy iteration algorithm, a dynamic programming method used for solving Markov Decision Processes (MDPs) with finite state and action spaces.


**Algorithm Overview**:

Policy iteration is an iterative algorithm that alternates between **policy evaluation** and **policy improvement** steps. It is guaranteed to converge to the optimal policy for MDPs with discrete states and actions. Here’s a breakdown of each step:

1. **Policy Evaluation**:
   
   - Given a policy ( :math:`\pi` ), the algorithm evaluates the expected value of each state under this policy. This involves solving for the value function \( :math:`V^\pi(s)` \) that satisfies:

     .. math::

        V^\pi(s) = \sum_a \pi(a|s) \sum_{s'} P(s'|s, a) \left[ R(s, a, s') + \gamma V^\pi(s') \right]

   - Where:
     
     - \( :math:`P(s'|s, a)` \) is the probability of transitioning from state \( :math:`s` \) to \( :math:`s'` \) given action \( :math:`a` \).
     
     - \( :math:`R(s, a, s')` \) is the reward received after transitioning from \( :math:`s` \) to \( :math:`s'` \).
     
     - \( :math:`\gamma` \) is the discount factor.

2. **Policy Improvement**:
   
   - Using the evaluated value function \( :math:`V^\pi` \), the algorithm updates the policy to be greedy with respect to \( :math:`V^\pi` \), selecting the action that maximizes the expected return:

     .. math::

        \pi'(s) = \arg\max_a \sum_{s'} P(s'|s, a) \left[ R(s, a, s') + \gamma V^\pi(s') \right]

   - If the policy no longer changes after an improvement step, the algorithm terminates, and the current policy is optimal.

The algorithm repeats these steps until the policy converges to the optimal policy, meaning further iterations do not change the policy.

************************************
Notes
************************************

**References**:

1. Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction (2nd ed.)*. MIT Press. Available at: http://incompleteideas.net/book/the-book-2nd.html

2. Bellman, R. E. (1957). *Dynamic Programming*. Princeton University Press.

3. Puterman, M. L. (1994). *Markov Decision Processes: Discrete Stochastic Dynamic Programming*. Wiley.

4. Bertsekas, D. P. (2007). *Dynamic Programming and Optimal Control (Vol. 1, 3rd ed.)*. Athena Scientific.

5. Howard, R. A. (1960). *Dynamic Programming and Markov Processes*. MIT Press.


.. note::
   By default, the agent uses a deterministic policy, meaning it selects the action with the highest probability. You can change this by setting the `policy` attribute to a custom policy function. Also note that the agent uses a discount factor (:math:`\gamma = 0.99`) by default, you can change this by setting the :math:`\gamma` attribute.

************************************
Can I use?
************************************

- Gym environments:

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

************************************
Example
************************************

.. code-block:: python

   from deeprl.agents import PolicyIterationAgent
   from deeprl.environments import GymnasiumEnvWrapper

   def main():

      env = GymnasiumEnvWrapper('FrozenLake-v1',is_slippery=False, render_mode='human')
      agent = PolicyIterationAgent(env)
      agent.learn()

      agent.interact(num_episodes=1, render=True)

   if __name__ == '__main__':
      main()


************************************
Parameters
************************************

.. autoclass:: deeprl.agents.policy_iteration_agent.PolicyIterationAgent
   :members:
   
************************************
See Also
************************************

- :class:`~deeprl.agents.value_iteration_agent.ValueIterationAgent` for an alternative dynamic programming method that uses value iteration to find the optimal policy.
- :class:`~deeprl.agents.q_learning_agent.QLearningAgent` for a model-free reinforcement learning agent that learns the optimal policy through Q-learning.
