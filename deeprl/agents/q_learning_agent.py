import torch
import json
from deeprl.agents.base_agent import Agent
from deeprl.policies.epsilon_greedy_policy import EpsilonGreedyPolicy
from deeprl.environments import GymnasiumEnvWrapper
from deeprl.visualization import ProgressBoard
from deeprl.reward_shaping.default_reward import DefaultReward
from deeprl.utils import print_progress

class QLearningAgent(Agent):
    """
    Q-learning agent using an epsilon-greedy policy with PyTorch tensors.
    
    :param env: The environment to interact with.
    :param learning_rate: The learning rate for updating the Q-table.
    :param discount_factor: The discount factor for future rewards.
    :param policy: The policy for selecting actions.
    :param is_continuous: Whether the environment has a continuous state space.
    :param approximator: The function approximator to use for continuous state spaces.
    :param reward_function: The reward function to use for the agent.
    :param verbose: Whether to display training progress.    
    """

    def __init__(self, 
                 env, 
                 learning_rate=0.1, 
                 discount_factor=0.99, 
                 policy=None, 
                 is_continuous=False,
                 approximator=None,
                 reward_shaping=None, 
                 verbose=False
    ):
        self.env = env
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.policy = policy or EpsilonGreedyPolicy(epsilon=0.1)
        self.is_continuous = is_continuous
        self.approximator = approximator
        self.reward_shaping = reward_shaping or DefaultReward()
        self.verbose = verbose
        
        
        if is_continuous and approximator is None:
            raise ValueError("An approximator must be provided for continuous state spaces.")
        
        if not is_continuous:
            self.q_table = torch.zeros((env.observation_space.n, env.action_space.n),       dtype=torch.float32)

    
    def act(self, state):
        if self.is_continuous:
            features = self.approximator.compute_features(state)
            q_values = torch.matmul(features, self.approximator.weights)
        else:
            q_values = self.q_table[state]
        return self.policy.select_action(q_values)
    
    def learn(self, episodes=1000, max_steps=100):
        """
        Train the agent by updating the Q-table.
        
        :param episodes: The number of episodes to train the agent.
        :param max_steps: The maximum number of steps per episode.
        :param save_train_graph: Whether to save a graph of the training progress.
        """
        episode_rewards = []

        # Display header for progress if verbose is enabled
        if self.verbose:
            print_progress(episode=0, total_reward=0, avg_reward=0, steps=0, epsilon=self.policy.epsilon, header=True)

        for episode in range(episodes):
            state = self.env.reset()
            total_reward, steps = 0, 0

            for _ in range(max_steps):
                if self.is_continuous:
                    state_features = self.approximator.compute_features(state)
                else:
                    state_features = state
                
                action = self.act(state)
                next_state, reward, done, truncated, info = self.env.step(action)
                
                reward = self.reward_shaping.shape(state, action, next_state, reward)
                
                self.update_q_table(state_features, action, reward, next_state, done)
                
                total_reward += reward
                state = next_state
                steps += 1
                if done or truncated:
                    break
                
            self.policy.update()
            episode_rewards.append(total_reward)
            

            avg_reward = sum(episode_rewards) / (episode + 1)
            if episode % 100 == 0 and self.verbose:
                print_progress(episode + 1, total_reward, avg_reward, steps, self.policy.epsilon)
                
        return episode_rewards

    def update_q_table(self, state_features, action, reward, next_state, done):
        """Update Q-table based on the agent's experience."""
        
        if self.is_continuous:
            
            next_features = self.approximator.compute_features(next_state)
            next_q_values = torch.matmul(next_features, self.approximator.weights)
            best_next_q = torch.max(next_q_values).item()
            target = reward + self.discount_factor * best_next_q * (1 - done)
                        
            self.approximator.weights[:, action] += (self.learning_rate * (target - self.predict(state_features, action)) * state_features).squeeze()
        else:
            
            best_next_action = torch.argmax(self.q_table[next_state]).item()
            target = reward + self.discount_factor * self.q_table[next_state][best_next_action] * (1 - done)
            self.q_table[state_features][action] += self.learning_rate * (target - self.q_table[state_features][action])
        
    def predict(self, state_features, action):
        """Predict the Q-value for a specific action in continuous state space."""
        
        return torch.matmul(state_features, self.approximator.weights[:, action])
    

    def save(self, filepath):
        """Save the complete agent state using PyTorch.
        
        :param filepath: The file path to save the agent's state.
        
        Saved fields:
        
        :param q_table: The Q-table to save (if not using function approximation).
        :param weights: The weights to save (if using function approximation).
        :param policy: The policy to save.
        :param reward_shaping: The reward shaping instance to save.
        :param learning_rate: The learning rate to save.
        :param discount_factor: The discount factor to save
        """
        state = {
            "is_continuous": self.is_continuous,
            "q_table": self.q_table if not self.is_continuous else None,
            "weights": self.approximator.weights if self.is_continuous else None,
            "policy": self.policy,  # Save policy instance
            "reward_shaping": self.reward_shaping,  # Save reward shaping instance
            "learning_rate": self.learning_rate,
            "discount_factor": self.discount_factor,
        }
        torch.save(state, filepath)
        print(f"Agent's state saved to {filepath}")

    def load(self, filepath):
        """Load the complete agent state using PyTorch."""
        state = torch.load(filepath)
        self.is_continuous = state["is_continuous"]
        if self.is_continuous:
            self.approximator.weights = state["weights"]
        else:
            self.q_table = state["q_table"]
        self.policy = state["policy"]
        self.reward_shaping = state["reward_shaping"]
        self.learning_rate = state["learning_rate"]
        self.discount_factor = state["discount_factor"]
        print(f"Agent's state loaded from {filepath}")
        
        
    def recreate_env_with_render(self):
        """
        Recreate the environment with render mode enabled.
        """
        env_name = self.env.spec.id
        env_kwargs = self.env.env_kwargs
        env_kwargs['render_mode'] = 'human'
        env = GymnasiumEnvWrapper(env=env_name, **env_kwargs)
        return env
           
        
    def get_env(self):
        return self.env
        