import pytest
import torch
from sklearn.kernel_approximation import RBFSampler
from deeprl.function_approximations.radial_basis_approximator import RBFBasisApproximator

@pytest.fixture
def mock_env():
    """Fixture to create a mock environment with Box observation and Discrete action spaces."""
    class MockEnv:
        class ObservationSpace:
            shape = (2,)  # Two-dimensional state space
        
        class ActionSpace:
            n = 3  # Three discrete actions
        
        observation_space = ObservationSpace()
        action_space = ActionSpace()
    
    return MockEnv()

@pytest.fixture
def approximator(mock_env):
    """Fixture to create an RBFBasisApproximator instance."""
    return RBFBasisApproximator(gamma=0.5, n_components=10, env=mock_env)

def test_initialization(approximator):
    """Test that the approximator initializes correctly."""
    assert approximator.weights.shape == (10, 3), "Weights shape should match RBF components and action space."
    assert isinstance(approximator.rbf, RBFSampler), "RBF sampler should be an instance of RBFSampler."

def test_compute_features(approximator):
    """Test that RBF features are computed correctly."""
    state = [0.5, -0.5]
    features = approximator.compute_features(state)
    assert features.shape == (1, 10), "Features shape should match the number of RBF components."
    assert features.dtype == torch.float32, "Features should be a PyTorch tensor with dtype float32."

def test_predict_all_actions(approximator):
    """Test prediction of Q-values for all actions."""
    state = [0.5, -0.5]
    predictions = approximator.predict(state)

    assert predictions.shape == (1,3), "Predictions should return Q-values for all actions."
    assert torch.equal(predictions, torch.zeros((1,3))), "Initial predictions should be zeros."

def test_predict_single_action(approximator):
    """Test prediction of Q-value for a single action."""
    state = [0.5, -0.5]
    action = 1
    prediction = approximator.predict(state, action)
    assert prediction == 0.0, "Initial prediction for a single action should be zero."

def test_update(approximator):
    """Test that the weights are updated correctly."""
    state = [0.5, -0.5]
    action = 2
    target = 1.0
    alpha = 0.1

    # Capture initial weights
    initial_weights = approximator.weights[:, action].clone()
    
    # Perform update
    approximator.update(state, target, action, alpha=alpha)

    # Check that weights have been updated
    assert not torch.equal(approximator.weights[:, action], initial_weights), "Weights for the action should be updated."

def test_update_error_direction(approximator):
    """Test that weights are updated in the direction of reducing error."""
    state = [0.5, -0.5]
    action = 1
    target = 1.0
    alpha = 0.1

    features = approximator.compute_features(state).squeeze()
    initial_prediction = approximator.predict(state, action)
    initial_weights = approximator.weights[:, action].clone()

    approximator.update(state, target, action, alpha=alpha)
    updated_prediction = approximator.predict(state, action)

    # Ensure prediction moves closer to target
    assert abs(updated_prediction - target) < abs(initial_prediction - target), \
        "Updated prediction should move closer to the target."

    # Ensure weights are adjusted correctly
    expected_update = alpha * (target - initial_prediction) * features
    assert torch.allclose(approximator.weights[:, action], initial_weights + expected_update), \
        "Weights should be updated correctly based on the error and features."

def test_invalid_action_index(approximator):
    """Test that an invalid action index raises an error."""
    state = [0.5, -0.5]
    invalid_action = 5  # Out of bounds for the action space
    target = 1.0

    with pytest.raises(IndexError):
        approximator.update(state, target, invalid_action, alpha=0.1)

