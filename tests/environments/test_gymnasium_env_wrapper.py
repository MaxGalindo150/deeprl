import pytest
import gymnasium as gym
from deeprl.environments.gymnasium_env_wrapper import GymnasiumEnvWrapper

@pytest.fixture
def env_wrapper():
    """Fixture to create a GymnasiumEnvWrapper instance with FrozenLake-v1 environment."""
    return GymnasiumEnvWrapper('FrozenLake-v1', is_slippery=False, render_mode='rgb_array')

def test_initialization(env_wrapper):
    """Test that the environment is initialized with correct observation and action spaces."""
    assert isinstance(env_wrapper.observation_space, gym.spaces.Space), "observation_space should be a Gym space."
    assert isinstance(env_wrapper.action_space, gym.spaces.Space), "action_space should be a Gym space."

def test_reset(env_wrapper):
    """Test that the reset method returns the initial state and works without errors."""
    state = env_wrapper.reset()
    assert state in range(env_wrapper.observation_space.n), "Reset state should be within the observation space."

def test_step(env_wrapper):
    """Test that the step method returns the expected output format."""
    env_wrapper.reset()
    action = env_wrapper.action_space.sample()  # Sample a random action
    next_state, reward, done, truncated, info = env_wrapper.step(action)
    
    assert next_state in range(env_wrapper.observation_space.n), "Next state should be within the observation space."
    assert isinstance(reward, (int, float)), "Reward should be a numeric type."
    assert isinstance(done, bool), "Done should be a boolean."
    assert isinstance(truncated, bool), "Truncated should be a boolean."
    assert isinstance(info, dict), "Info should be a dictionary."

def test_render(env_wrapper):
    """Test that the render method works without errors after resetting the environment."""
    try:
        env_wrapper.reset()  # Ensure the environment is reset before rendering
        env_wrapper.render()
    except Exception as e:
        pytest.fail(f"Render method raised an exception: {e}")


def test_close(env_wrapper):
    """Test that the close method works without errors."""
    try:
        env_wrapper.close()
    except Exception as e:
        pytest.fail(f"Close method raised an exception: {e}")

def test_get_underlying_env(env_wrapper):
    """Test that get_underlying_env returns the base Gymnasium environment."""
    underlying_env = env_wrapper.get_underlying_env()
    assert isinstance(underlying_env, gym.Env), "Underlying environment should be an instance of Gym Env."
    assert hasattr(underlying_env, 'P'), "Underlying environment should have attribute 'P' for FrozenLake."
