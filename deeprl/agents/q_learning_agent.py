from deeprl.agents import Agent
from deeprl.policies import EpsilonGreedyPolicy
import numpy as np
import json

class QLearningAgent(Agent):
    """
    Q-learning agent that uses an epsilon-greedy policy.
    """

    def __init__(self, env, learning_rate=0.1, discount_factor=0.99, epsilon=0.1, policy=None):
        """
        Initialize the Q-learning agent.

        :param env: The environment.
        :param learning_rate: The learning rate (alpha).
        :param discount_factor: The discount factor (gamma).
        :param epsilon: Initial exploration rate for epsilon-greedy policy.
        """
        self.env = env
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor

        # Initialize Q-table with zeros
        self.q_table = np.zeros((env.observation_space.n, env.action_space.n))

        # Use EpsilonGreedyPolicy for action selection
        self.policy = policy if policy else EpsilonGreedyPolicy(epsilon)
    
    def act(self, state):
        """
        Select an action based on the epsilon-greedy policy.

        :param state: The current state of the environment.
        :return: The selected action.
        """
        q_values = self.q_table[state]
        return self.policy.select_action(q_values)
    
    def learn(self, state, action, reward, next_state, done):
        """
        Update the Q-table based on the agent's experience.

        :param state: The current state.
        :param action: The action taken.
        :param reward: The reward received after taking the action.
        :param next_state: The state reached after taking the action.
        :param done: Whether the episode is done.
        """
        best_next_action = np.argmax(self.q_table[next_state])
        target = reward + self.discount_factor * self.q_table[next_state][best_next_action] * (1 - done)
        self.q_table[state][action] += self.learning_rate * (target - self.q_table[state][action])
    
    def save(self, filepath):
        """
        Save the Q-table to a file in JSON format.

        :param filepath: The path to the file.
        """
        with open(filepath, 'w') as f:
            json.dump(self.q_table.tolist(), f)
        print(f"Q-table saved to {filepath}")

    def load(self, filepath):
        """
        Load the Q-table from a file.

        :param filepath: The path to the file.
        """
        with open(filepath, 'r') as f:
            self.q_table = np.array(json.load(f))
        print(f"Q-table loaded from {filepath}")
    
    def interact(self, env, episodes=1, max_steps=100):
        """
        Interact with the environment for a specific number of episodes.

        :param env: The environment to interact with.
        :param episodes: Number of episodes to run.
        :param max_steps: Maximum steps per episode.
        :return: List of cumulative rewards per episode.
        """
        episode_rewards = []

        for episode in range(episodes):
            state = env.reset()
            done = False
            total_reward = 0

            for step in range(max_steps):
                action = self.act(state)
                next_state, reward, done, truncated, info = env.step(action)
                
                # Update Q-table with new experience
                self.learn(state, action, reward, next_state, done)
                
                total_reward += reward
                state = next_state
                
                if done:
                    break
            
            episode_rewards.append(total_reward)
            if episode % 10 == 0:
                print(f"Episode {episode + 1}/{episodes}: Total Reward = {total_reward}")

        return episode_rewards
