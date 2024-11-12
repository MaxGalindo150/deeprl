import json
import numpy as np
import pickle

from deeprl.agents import Agent
from deeprl.policies import DeterministicPolicy

class ValueIterationAgent(Agent):
    """
    Agent that uses Value Iteration to learn the optimal policy.
    
    
    :param env: The environment.
    :param gamma: The discount factor.
    :param theta: The convergence threshold.
    :param policy: The policy to use (default is a deterministic policy).
    """
    
    def __init__(self, env, gamma=0.99, theta=1e-5, policy=None):
        """
        Initialize the agent.
        """
        self.env = env
        self.gamma = gamma
        self.theta = theta
        self.V = np.zeros(env.observation_space.n)
        self.policy = policy if policy else DeterministicPolicy(env.observation_space)
        
    def value_iteration(self):
        """"
        Execute the Value Iteration algorithm.
        """
        while True:
            delta = 0
            for state in range(self.env.observation_space.n):
                v = self.V[state]
                q_values = self.compute_q_values(state)
                self.V[state] = max(q_values)
                delta = max(delta, abs(v - self.V[state]))
            if delta < self.theta:
                break
        self.update_policy()
        
    
    def compute_q_values(self, state):
        """
        Compute the Q-values for all actions in a given state.
        
        :param state: The state.
        :return: List of Q-values.
        """
        underlying_env = self.env.get_underlying_env()  # Get the underlying environment
        q_values = np.zeros(underlying_env.action_space.n)

        for action in range(underlying_env.action_space.n):
            if hasattr(underlying_env, 'P'):  # Check if the environment has a transition matrix
                for prob, next_state, reward, done in underlying_env.P[state][action]:
                    q_values[action] += prob * (reward + self.gamma * self.V[next_state])
            else:
                raise AttributeError("The environment does not have a transition matrix.")
        
        return q_values
    
    def update_policy(self):
        """
        Update the policy based on the Value function.
        """
        for state in range(self.env.observation_space.n):
            q_values = self.compute_q_values(state)
            self.policy.update(state, np.argmax(q_values).item())
    
    def act(self, state):
        """
        Select an action based on the state and the current policy.
        
        :param state: The current state of the environment.
        :return: The selected action.
        """
        # Be sure that the state is an integer
        if not isinstance(state, int):
            raise TypeError(f"Expected state to be an int, but got {type(state)}")

        return self.policy.select_action(state)

    
    def learn(self):
        """"
        Update the agent's parameters based on the collected experience.
        """
        self.value_iteration()
        
    def save(self, filepath):
        """
        Save the agent's parameters (value table and policy) to a JSON file.
        
        :param filepath: The path to the file.
        """
        policy_dict = {state: int(action) for state, action in enumerate(self.policy.policy_table)}

        data = {
            'value_table': self.V.tolist(),  # Convert numpy array to list
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
        self.V = np.array(data['value_table'])
        self.policy.policy_table = np.array([data['policy'][str(state)] for state in range(len(data['policy']))])
        print(f"Agent's parameters loaded from {filepath}")
    
    def get_env(self):
        return self.env
