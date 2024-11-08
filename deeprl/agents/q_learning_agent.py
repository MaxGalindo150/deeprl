import torch
import json
from deeprl.agents.base_agent import Agent
from deeprl.policies.epsilon_greedy_policy import EpsilonGreedyPolicy
from deeprl.environments import GymnasiumEnvWrapper
from deeprl.visualization import ProgressBoard
from deeprl.utils import print_progress

class QLearningAgent(Agent):
    """
    Q-learning agent using an epsilon-greedy policy with PyTorch tensors.
    
    :param env: The environment to interact with.
    :param learning_rate: The learning rate for updating the Q-table.
    :param discount_factor: The discount factor for future rewards.
    :param epsilon: The probability of selecting a random action during training.
    :param step_penalty: The penalty for each step taken in the environment.
    :param verbose: Whether to display training progress.
    
    """

    def __init__(self, env, learning_rate=0.1, discount_factor=0.99, epsilon=0.1, step_penalty=0.0, verbose=False):
        self.env = env
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.step_penalty = step_penalty
        self.verbose = verbose
        self.q_table = torch.zeros((env.observation_space.n, env.action_space.n), dtype=torch.float32)
        self.policy = EpsilonGreedyPolicy(epsilon=epsilon)
    
    def act(self, state):
        return self.policy.select_action(self.q_table[state])
    
    def learn(self, episodes=1000, max_steps=100, save_train_graph=False):
        """Train the agent by updating the Q-table."""
        episode_rewards = []
        progress_board = ProgressBoard(xlabel="Episode", ylabel="Cumulative Reward", save_path="q_learning_training.png")

        # Display header for progress if verbose is enabled
        if self.verbose:
            print_progress(episode=0, total_reward=0, avg_reward=0, steps=0, header=True)

        for episode in range(episodes):
            state = self.env.reset()
            total_reward, steps = 0, 0

            for _ in range(max_steps):
                action = self.act(state)
                next_state, reward, done, truncated, info = self.env.step(action)
                
                reward = self.step_penalty if reward == 0 and self.step_penalty != 0 else reward
                self.update_q_table(state, action, reward, next_state, done)
                
                total_reward += reward
                state = next_state
                steps += 1
                if done:
                    break

            episode_rewards.append(total_reward)
            
            if save_train_graph:
                progress_board.record(total_reward)

            # Print progress every 10 episodes if verbose is enabled
            avg_reward = sum(episode_rewards) / (episode + 1)
            if episode % 10 == 0 and self.verbose:
                print_progress(episode + 1, total_reward, avg_reward, steps)
        if save_train_graph:
            progress_board.save()
        return episode_rewards

    def update_q_table(self, state, action, reward, next_state, done):
        """Update Q-table based on the agent's experience."""
        best_next_action = torch.argmax(self.q_table[next_state]).item()
        target = reward + self.discount_factor * self.q_table[next_state][best_next_action] * (1 - done)
        self.q_table[state][action] += self.learning_rate * (target - self.q_table[state][action])
    
    def interact(self, episodes=1, max_steps=100, render=False, save_test_graph=False):
        """Evaluate the agent in the environment without updating Q-table."""
        episode_rewards = []
        progress_board = ProgressBoard(xlabel="Episode", ylabel="Cumulative Reward", save_path="q_learning_test.png")
        env = GymnasiumEnvWrapper('FrozenLake-v1', is_slippery=False, render_mode='human') if render else self.env

        for episode in range(episodes):
            state = env.reset()
            total_reward = 0

            for _ in range(max_steps):
                if render:
                    env.render()
                action = torch.argmax(self.q_table[state]).item()
                next_state, reward, done, truncated, info = env.step(action)
                
                total_reward += reward
                state = next_state
                if done:
                    break

            episode_rewards.append(total_reward)
            if save_test_graph:
                progress_board.record(total_reward)
                
        if save_test_graph:
            progress_board.save()

        if render:
            env.close()

        return episode_rewards

    def save(self, filepath):
        """Save the Q-table to a file."""
        with open(filepath, 'w') as f:
            json.dump(self.q_table.tolist(), f)
        print(f"Q-table saved to {filepath}")

    def load(self, filepath):
        """Load the Q-table from a file."""
        with open(filepath, 'r') as f:
            q_table_list = json.load(f)
            self.q_table = torch.tensor(q_table_list, dtype=torch.float32)
        print(f"Q-table loaded from {filepath}")
