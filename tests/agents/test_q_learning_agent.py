import pytest
import torch
import os
from deeprl.agents.q_learning_agent import QLearningAgent
from deeprl.policies.epsilon_greedy_policy import EpsilonGreedyPolicy
from deeprl.environments import GymnasiumEnvWrapper

@pytest.fixture
def frozen_lake_env():
    """Fixture to create the FrozenLake environment."""
    return GymnasiumEnvWrapper('FrozenLake-v1', is_slippery=False)

@pytest.fixture
def q_learning_agent(frozen_lake_env):
    """Fixture to create a QLearningAgent instance."""
    return QLearningAgent(env=frozen_lake_env, learning_rate=0.1, discount_factor=0.99, verbose=False)

def test_initialization(q_learning_agent, frozen_lake_env):
    """Test agent initialization."""
    assert q_learning_agent.env == frozen_lake_env
    assert q_learning_agent.learning_rate == 0.1
    assert q_learning_agent.discount_factor == 0.99
    assert isinstance(q_learning_agent.policy, EpsilonGreedyPolicy)
    assert q_learning_agent.q_table.shape == (frozen_lake_env.observation_space.n, frozen_lake_env.action_space.n)

def test_act(q_learning_agent):
    """Test action selection."""
    state = 0
    action = q_learning_agent.act(state)
    assert 0 <= action < q_learning_agent.env.action_space.n

def test_update_q_table(q_learning_agent):
    """Test Q-table update logic."""
    state, action, reward, next_state, done = 0, 1, 1.0, 2, False
    initial_q_value = q_learning_agent.q_table[state][action].item()
    
    q_learning_agent.update_q_table(state, action, reward, next_state, done)
    updated_q_value = q_learning_agent.q_table[state][action].item()
    
    assert updated_q_value != initial_q_value
    assert updated_q_value > initial_q_value

def test_learn(q_learning_agent):
    """Test the training process."""
    rewards = q_learning_agent.learn(episodes=10, max_steps=50)
    assert len(rewards) == 10
    assert all(isinstance(r, (float, int)) for r in rewards)

def test_interact(q_learning_agent):
    """Test the agent's interaction with the environment."""
    rewards = q_learning_agent.interact(episodes=5, max_steps=50)
    assert len(rewards) == 5
    assert all(isinstance(r, (float, int)) for r in rewards)

def test_save_and_load(q_learning_agent, tmp_path):
    """Test saving and loading the Q-table."""
    filepath = tmp_path / "q_table.json"
    q_learning_agent.save(filepath)
    
    assert os.path.exists(filepath)
    
    initial_q_table = q_learning_agent.q_table.clone()
    q_learning_agent.q_table.fill_(0)
    q_learning_agent.load(filepath)
    
    assert torch.equal(q_learning_agent.q_table, initial_q_table)
