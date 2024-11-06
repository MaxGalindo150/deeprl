import numpy as torch
import pickle

from deeprl.agents import Agent
from deeprl.policies import DeterministicPolicy

import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ValueIterationAgent(Agent):
    """
    Agent that uses Value Iteration to learn the optimal policy.
    """
    
    def __init__(self, env, gamma=0.99, theta=1e-5, policy=None):
        """
        Initialize the agent.
        d
        :param env: The environment.
        :param gamma: The discount factor.
        :param theta: The convergence threshold.
        """
        self.env = env
        self.gamma = gamma
        self.theta = theta
        self.V = torch.zeros(env.observation_space.n)
        self.policy = policy if policy else DeterministicPolicy(env.observation_space)
        
    def value_iteration(self):
        """"
        Execute the Value Iteration algorithm.
        """
        iteration = 0
        while True:
            delta = 0
            for state in range(self.env.observation_space.n):
                v = self.V[state]
                q_values = self.compute_q_values(state)
                self.V[state] = max(q_values)
                delta = max(delta, abs(v - self.V[state]))
            iteration += 1
            if delta < self.theta:
                logger.info(f"Value Iteration converged after {iteration} iterations.")
                break
        self.update_policy()
        
    
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
            self.policy.update_policy(state, torch.argmax(q_values).item())
    
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
                
                action = int(self.act(state).item())
                next_state, reward, done, truncated, info = self.env.step(action)
                total_reward += reward
                state = next_state
            
            episode_rewards.append(total_reward)
            self.env.close()
            if render:
                print(f"Episode {episode + 1}: Total Reward = {total_reward}")
        
        return episode_rewards
