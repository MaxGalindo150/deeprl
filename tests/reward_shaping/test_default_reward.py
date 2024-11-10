import pytest
from deeprl.reward_shaping.default_reward import DefaultReward

@pytest.fixture
def default_reward():
    """Fixture to create a DefaultReward instance."""
    return DefaultReward()

def test_shape_no_modification(default_reward):
    """
    Test that the reward remains unchanged.
    """
    state = [0.5, -0.5]  # Example state
    action = 1  # Example action
    next_state = [0.6, -0.4]  # Example next state
    reward = 10.0  # Example reward
    
    shaped_reward = default_reward.shape(state, action, next_state, reward)
    
    assert shaped_reward == reward, "DefaultReward should not modify the reward."

def test_shape_with_zero_reward(default_reward):
    """
    Test that a zero reward remains zero.
    """
    state = [0.0, 0.0]  # Example state
    action = 0  # Example action
    next_state = [0.0, 0.0]  # Example next state
    reward = 0.0  # Zero reward
    
    shaped_reward = default_reward.shape(state, action, next_state, reward)
    
    assert shaped_reward == reward, "DefaultReward should not modify a zero reward."

def test_shape_negative_reward(default_reward):
    """
    Test that a negative reward remains unchanged.
    """
    state = [0.5, -0.5]  # Example state
    action = 2  # Example action
    next_state = [0.6, -0.6]  # Example next state
    reward = -5.0  # Negative reward
    
    shaped_reward = default_reward.shape(state, action, next_state, reward)
    
    assert shaped_reward == reward, "DefaultReward should not modify a negative reward."

def test_shape_large_reward(default_reward):
    """
    Test that a large reward remains unchanged.
    """
    state = [0.1, 0.2]  # Example state
    action = 1  # Example action
    next_state = [0.2, 0.3]  # Example next state
    reward = 1e6  # Large reward
    
    shaped_reward = default_reward.shape(state, action, next_state, reward)
    
    assert shaped_reward == reward, "DefaultReward should not modify a large reward."
