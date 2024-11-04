import numpy as np
import pickle

from deeprl.agents.base_agent import Agent

class ValueIterationAgent(Agent):
    """
    Agent that uses Value Iteration to learn the optimal policy.
    """
    
    def __init__(self, env, gamma=0.99, theta=1e-5):
        """
        Initialize the agent.
        
        :param env: The environment.
        :param gamma: The discount factor.
        :param theta: The convergence threshold.
        """
        self.env = env
        self.gamma = gamma
        self.theta = theta
        self.V = np.zeros(env.observation_space.n)
        self.policy = np.zeros(env.observation_space.n, dtype=int)
        
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
        underlying_env = self.env.get_underlying_env()  # Accede al entorno interno
        q_values = np.zeros(underlying_env.action_space.n)

        for action in range(underlying_env.action_space.n):
            if hasattr(underlying_env, 'P'):  # Verifica si el entorno tiene el atributo `P`
                for prob, next_state, reward, done in underlying_env.P[state][action]:
                    q_values[action] += prob * (reward + self.gamma * self.V[next_state])
            else:
                raise AttributeError("El entorno no tiene un atributo `P` para acceder a la matriz de transición.")
        
        return q_values
    
    def update_policy(self):
        """
        Update the policy based on the Value function.
        """
        for state in range(self.env.observation_space.n):
            q_values = self.compute_q_values(state)
            self.policy[state] = np.argmax(q_values)
    
    def act(self, state):
        """
        Select an action based on the state and the current policy.
        
        :param state: The current state of the environment.
        :return: The selected action.
        """
        # Asegúrate de que el estado sea un número entero  
        if not isinstance(state, int):
            raise TypeError(f"Expected state to be an int, but got {type(state)}")

        return self.policy[state]

    
    def learn(self):
        """"
        Update the agent's parameters based on the collected experience.
        """
        self.value_iteration()
        
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
        