import pytest
import numpy as np
from unittest.mock import MagicMock

from deeprl.common.evaluate_policy import evaluate_policy

@pytest.fixture
def test_evaluate_policy():
    """Test the evaluate_policy function."""

    # Mock the environment
    class MockEnv:
        def __init__(self):
            self.current_step = 0
            self.max_steps = 5
            self.done = False

        def reset(self):
            self.current_step = 0
            self.done = False
            return [0.0, 0.0]  # Mock state

        def step(self, action):
            self.current_step += 1
            reward = 1.0  # Fixed reward
            self.done = self.current_step >= self.max_steps
            return [0.0, 0.0], reward, self.done, False, {}

    # Mock the model
    class MockModel:
        def act(self, state):
            return 0  # Always return the same action

    # Instantiate mocks
    mock_env = MockEnv()
    mock_model = MockModel()

    # Test evaluate_policy
    num_eval_episodes = 3
    mean_reward, std_reward = evaluate_policy(mock_model, mock_env, num_eval_episodes)

    # Assertions
    assert mean_reward == 5.0, "Mean reward should be equal to the maximum reward per episode."
    assert std_reward == 0.0, "Standard deviation should be zero if all episodes yield the same reward."
