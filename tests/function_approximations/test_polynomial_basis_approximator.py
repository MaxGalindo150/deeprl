import pytest
import torch
import numpy as np
from gymnasium.spaces import Box, Discrete
from sklearn.preprocessing import PolynomialFeatures
from deeprl.function_approximations.polynomial_basis_approximator import PolynomialBasisApproximator

@pytest.fixture
def mock_env():
    """Fixture to create a mock environment with Box observation and Discrete action spaces."""
    class MockEnv:
        observation_space = Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        action_space = Discrete(3)
        
        def reset(self):
            return np.zeros(self.observation_space.shape, dtype=self.observation_space.dtype)
        
        def step(self, action):
            # Dummy next state, reward, done, truncated, info
            next_state = np.random.uniform(
                self.observation_space.low, self.observation_space.high, size=self.observation_space.shape
            ).astype(self.observation_space.dtype)
            reward = 1.0
            done = False
            truncated = False
            info = {}
            return next_state, reward, done, truncated, info
    return MockEnv()

@pytest.fixture
def approximator(mock_env):
    """Fixture to create a PolynomialBasisApproximator instance."""
    return PolynomialBasisApproximator(degree=2, env=mock_env)

def test_initialization(mock_env):
    """Test that the approximator initializes correctly."""
    approximator = PolynomialBasisApproximator(degree=2, env=mock_env)
    dummy_state = torch.zeros(mock_env.observation_space.shape[0]).reshape(1, -1).numpy()
    poly = PolynomialFeatures(degree=2)
    poly.fit_transform(dummy_state)

    assert approximator.weights.shape == (poly.n_output_features_, mock_env.action_space.n)
    assert torch.equal(approximator.weights, torch.zeros_like(approximator.weights))

def test_compute_features(approximator):
    """Test that the computed features match the expected polynomial transformation."""
    state = [0.5, -0.5]
    features = approximator.compute_features(state)
    
    poly = PolynomialFeatures(degree=2)
    expected_features = poly.fit_transform(torch.tensor(state).reshape(1, -1).numpy())
    
    assert torch.allclose(features, torch.tensor(expected_features, dtype=torch.float32))

def test_predict(approximator):
    """Test that the prediction is computed correctly."""
    state = [0.5, -0.5]
    approximator.weights += 0.5  # Assign some weights
    prediction = approximator.predict(state)
    
    features = approximator.compute_features(state)
    expected_prediction = torch.matmul(features, approximator.weights)
    
    assert torch.allclose(prediction, expected_prediction)

# def test_update(approximator):
#     state = [0.5, -0.5]
#     target = torch.tensor([1.0, 0.5, -0.5])  # Mock target values for 3 actions
#     alpha = 0.1

#     initial_weights = approximator.weights.clone()
#     approximator.update(state, target, alpha=alpha)
    
#     assert not torch.equal(approximator.weights, initial_weights), "Weights should be updated."


def test_invalid_initialization():
    """Test that initializing without an environment raises an error."""
    with pytest.raises(ValueError, match="An environment instance is required to initialize the approximator."):
        PolynomialBasisApproximator(degree=2)
