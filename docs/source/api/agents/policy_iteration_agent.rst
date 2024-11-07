PolicyIterationAgent
=====================
The `PolicyIterationAgent` class implements the policy iteration algorithm, a dynamic programming method used for solving Markov Decision Processes (MDPs) with finite state and action spaces.

.. autoclass:: deeprl.agents.policy_iteration_agent.PolicyIterationAgent
   :members:
   :undoc-members:
   :show-inheritance:

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

**Example Usage**:
Here's how to use `PolicyIterationAgent`:

.. code-block:: python

   from deeprl.agents import PolicyIterationAgent
   from deeprl.environments import GymnasiumEnvWrapper

   def main():
      # Configure the FrozenLake environment with a slippery surface using GymnasiumEnvWrapper from DeepRL
      env = GymnasiumEnvWrapper('FrozenLake-v1',is_slippery=False, render_mode='human')
      agent = PolicyIterationAgent(env)
      agent.learn()

      # Unpack the initial state and reset the environment
      agent.interact(num_episodes=1, render=True)

   if __name__ == '__main__':
      main()

**Method Summary**:

- `learn(self)`: Runs the policy iteration algorithm to determine the optimal policy. This method alternates between policy evaluation and policy improvement until convergence.

**Details**:

- **Policy Evaluation**: This step iteratively calculates the state values for the current policy until the value function converges.

- **Policy Improvement**: Once the values have converged, the policy is updated to select the action that maximizes the expected return based on the current value function.

- **Policy**: By default, the agent uses a determinist policy, meaning it selects the action with the highest probability, the policy is improved by selecting the action that maximizes the expected return. You can change this by setting the `policy` attribute to a custom policy function.   

**Convergence**:

- Policy iteration is guaranteed to converge to the optimal policy when applied to MDPs with discrete and finite state and action spaces.

**See Also**: 
- `deeprl.agents.ValueIterationAgent` for an alternative dynamic programming method that uses value iteration to find the optimal policy.