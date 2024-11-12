import json
import numpy as np
import pickle

from deeprl.agents.base_agent import Agent
from deeprl.policies import DeterministicPolicy

class PolicyIterationAgent(Agent):
    """
    Agent that implements the Policy Iteration algorithm.
    
    :param env: The environment.
    :param gamma: The discount factor.
    :param theta: The convergence threshold.
    :param policy: The policy to use (default is a deterministic policy).
    """

    def __init__(self, env, gamma=0.99, theta=1e-6, policy=None):
        
        self.env = env
        self.gamma = gamma
        self.theta = theta
        self.policy = policy if policy else DeterministicPolicy(observation_space=env.observation_space)
        self.value_table = np.zeros(env.observation_space.n)

    def policy_evaluation(self):
    
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
        
        policy_stable = True
        for state in range(self.env.observation_space.n):
            old_action = self.policy.select_action(state)
            q_values = self.compute_q_values(state)
            self.policy.update(state, np.argmax(q_values).item())
            if old_action != self.policy.select_action(state):
                policy_stable = False
        return policy_stable

    def compute_q_values(self, state):
        
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
        
        while True:
            self.policy_evaluation()
            if self.update_policy():
                break

    def act(self, state):
        
        return self.policy.select_action(state)

    def learn(self):
        
        self.policy_iteration()


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
        
        
    def get_env(self):
        return self.env