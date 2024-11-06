import torch
import pickle

from deeprl.agents.base_agent import Agent
from deeprl.policies import DeterministicPolicy

class PolicyIterationAgent(Agent):
    """
    Agent that implements the Policy Iteration algorithm.
    """

    def __init__(self, env, gamma=0.99, theta=1e-6, policy=None):
        """
        Initialize the PolicyIterationAgent.
        This method initializes the PolicyIterationAgent with the given environment, discount factor, and convergence threshold. The agent will use these parameters to perform policy iteration to find the optimal policy.
        :param env: The environment in which the agent will operate. This should be an instance of a Gymnasium environment or a compatible environment wrapper.
        :type env: gymnasium.Env or GymnasiumEnvWrapper
        :param gamma: The discount factor for future rewards. This value should be between 0 and 1, where 0 means only immediate rewards are considered, and 1 means future rewards are fully considered.
        :type gamma: float, optional
        :param theta: The convergence threshold for policy evaluation. The iteration stops when the value function change is less than this threshold.
        :type theta: float, optional
        """
        self.env = env
        self.gamma = gamma
        self.theta = theta
        self.policy = policy if policy else DeterministicPolicy(observation_space=env.observation_space)
        self.value_table = torch.zeros(env.observation_space.n)
    def policy_evaluation(self):
        """
        Evaluate the current policy using the value iteration algorithm.

        This method iteratively updates the value function for each state under the current policy until the value function converges. The convergence is determined by the threshold `theta`.

        The value function is updated using the Bellman equation for the given policy:
        
            V(s) = sum(P(s'|s,a) * [R(s,a,s') + gamma * V(s')])

        where:
        - V(s) is the value of state s.
        - P(s'|s,a) is the probability of transitioning to state s' from state s given action a.
        - R(s,a,s') is the reward received after transitioning from state s to state s' given action a.
        - gamma is the discount factor for future rewards.

        :return: None
        """
        underlying_env = self.env.get_underlying_env()
        while True:
            delta = 0
            for state in range(underlying_env.observation_space.n):
                v = self.value_table[state]
                action = self.policy.select_action(state)
                self.value_table[state] = sum([
                    prob * (reward + self.gamma * self.value_table[next_state])
                    for prob, next_state, reward, done in underlying_env.P[state][action]
                ])
                delta = max(delta, abs(v - self.value_table[state]))
            if delta < self.theta:
                break

    def update_policy(self):
        """
        Improve the current policy using the value table. Also known as policy improvement.
        """
        policy_stable = True
        for state in range(self.env.observation_space.n):
            old_action = self.policy.select_action(state)
            q_values = self.compute_q_values(state)
            self.policy.update_policy(state, torch.argmax(q_values).item())
            if old_action != self.policy.select_action(state):
                policy_stable = False
        return policy_stable

    def compute_q_values(self, state):
        """
        Compute the Q-values for all actions in a given state.
        
        :param state: The state.
        :return: List of Q-values.
        """
        underlying_env = self.env.get_underlying_env()  # Get the underlying environment
        q_values = torch.zeros(underlying_env.action_space.n)

        for action in range(underlying_env.action_space.n):
            if hasattr(underlying_env, 'P'):  # Check if the environment has a transition matrix
                for prob, next_state, reward, done in underlying_env.P[state][action]:
                    q_values[action] += prob * (reward + self.gamma * self.value_table[next_state])
            else:
                raise AttributeError("The environment does not have a transition matrix.")
        
        return q_values

    def policy_iteration(self):
        """
        Execute the Policy Iteration
        """
        while True:
            self.policy_evaluation()
            if self.update_policy():
                break

    def act(self, state):
        """
        Selecciona la acción basada en la política actual.
        """
        return self.policy.select_action(state)

    def learn(self):
        """
        Ejecuta el proceso de aprendizaje (iteración de política).
        """
        self.policy_iteration()

    def interact(self, num_episodes=1, render=False):
        """
        Interact with the environment following the learned policy for a given number of episodes.
        
        :param num_episodes: Number of episodes to run.
        :param render: If True, render the environment during interaction.
        :return: List of total rewards obtained in each episode.
        """
        episode_rewards = []
        
        for episode in range(num_episodes):
            state = self.env.reset()
            done = False
            total_reward = 0
            
            while not done:
                if render:
                    self.env.render()
                
                action = int(self.act(state))
                next_state, reward, done, truncated, info = self.env.step(action)
                total_reward += reward
                state = next_state
            
            episode_rewards.append(total_reward)
            self.env.close()
            if render:
                print(f"Episode {episode + 1}: Total Reward = {total_reward}")
        
        return episode_rewards
    
    def save(self, filepath):
        """
        Save the agent's parameters (V and policy) to a file.
        
        :param filepath: The path to the file.
        """
        data = {
            'V': self.V,
            'policy': self.policy
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        print(f"Agent's parameters saved to {filepath}")

    def load(self, filepath):
        """
        Load the agent's parameters (V and policy) from a file.
        
        :param filepath: The path to the file.
        """
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        self.V = data['V']
        self.policy = data['policy']
        print(f"Agent's parameters loaded from {filepath}")