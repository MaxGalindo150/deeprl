import json
import numpy as np
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
        """
        self.env = env
        self.gamma = gamma
        self.theta = theta
        self.policy = policy if policy else DeterministicPolicy(observation_space=env.observation_space)
        self.value_table = np.zeros(env.observation_space.n)

    def policy_evaluation(self):
        """
        Evaluate the current policy using the value iteration algorithm.
        """
        underlying_env = self.env.get_underlying_env()
        while True:
            delta = 0
            for state in range(underlying_env.observation_space.n):
                v = self.value_table[state]
                action = self.policy.select_action(state)
                self.value_table[state] = sum(
                    prob * (reward + self.gamma * self.value_table[next_state])
                    for prob, next_state, reward, done in underlying_env.P[state][action]
                )
                delta = max(delta, abs(v - self.value_table[state]))
            if delta < self.theta:
                break

    def update_policy(self):
        """
        Improve the current policy using the value table.
        """
        policy_stable = True
        for state in range(self.env.observation_space.n):
            old_action = self.policy.select_action(state)
            q_values = self.compute_q_values(state)
            self.policy.update_policy(state, np.argmax(q_values).item())
            if old_action != self.policy.select_action(state):
                policy_stable = False
        return policy_stable

    def compute_q_values(self, state):
        """
        Compute the Q-values for all actions in a given state.
        
        :param state: The state.
        :return: List of Q-values.
        """
        underlying_env = self.env.get_underlying_env()
        q_values = np.zeros(underlying_env.action_space.n)

        for action in range(underlying_env.action_space.n):
            if hasattr(underlying_env, 'P'):
                for prob, next_state, reward, done in underlying_env.P[state][action]:
                    q_values[action] += prob * (reward + self.gamma * self.value_table[next_state])
            else:
                raise AttributeError("The environment does not have a transition matrix.")
        
        return q_values

    def policy_iteration(self):
        """
        Execute the Policy Iteration algorithm.
        """
        while True:
            self.policy_evaluation()
            if self.update_policy():
                break

    def act(self, state):
        """
        Select an action based on the state and the current policy.
        
        :param state: The current state of the environment.
        :return: The selected action.
        """
        return self.policy.select_action(state)

    def learn(self):
        """
        Execute the learning process (policy iteration).
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
        Save the agent's parameters (value table and policy) to a JSON file.
        
        :param filepath: The path to the file.
        """
        policy_dict = {state: int(action) for state, action in enumerate(self.policy.policy_table)}

        data = {
            'value_table': self.value_table.tolist(),  # Convert numpy array to list
            'policy': policy_dict
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=4)
        print(f"Agent's parameters saved to {filepath}")

    def load(self, filepath):
        """
        Load the agent's parameters (value table and policy) from a JSON file.
        
        :param filepath: The path to the file.
        """
        with open(filepath, 'r') as f:
            data = json.load(f)
        self.value_table = np.array(data['value_table'])
        self.policy.policy_table = np.array([data['policy'][str(state)] for state in range(len(data['policy']))])
        print(f"Agent's parameters loaded from {filepath}")