import pytest
import torch
import numpy as np
from deeprl.policies.epsilon_greedy_policy import EpsilonGreedyPolicy

@pytest.fixture
def policy():
    """Fixture to create an EpsilonGreedyPolicy instance."""
    return EpsilonGreedyPolicy(epsilon=0.1)

def test_policy_initialization(policy):
    """Test that the policy initializes with the correct epsilon value."""
    assert policy.epsilon == 0.1, "Epsilon should initialize to 0.1."

def test_select_action_exploit(policy, mocker):
    """Test that the policy selects the action with the highest Q-value (exploit)."""
    mocker.patch("torch.rand", return_value=torch.tensor(0.2))  # Ensure epsilon condition is not met
    q_values = torch.tensor([1.0, 2.0, 3.0])  # Highest Q-value at index 2
    action = policy.select_action(q_values)
    assert action == 2, "The policy should select the action with the highest Q-value."

def test_select_action_explore(policy, mocker):
    """Test that the policy selects a random action (explore)."""
    mocker.patch("torch.rand", return_value=torch.tensor(0.05))  # Ensure epsilon condition is met
    q_values = torch.tensor([1.0, 2.0, 3.0])
    random_action = policy.select_action(q_values)
    assert random_action in range(len(q_values)), "The action should be a valid random choice."

def test_set_epsilon(policy):
    """Test updating the epsilon value."""
    policy.set_epsilon(0.5)
    assert policy.epsilon == 0.5, "Epsilon should update to the new value."

def test_exploit_probability(policy, mocker):
    """Test the probability of exploitation."""
    policy.set_epsilon(0.0)  # Always exploit
    q_values = torch.tensor([1.0, 2.0, 3.0])
    action = policy.select_action(q_values)
    assert action == 2, "With epsilon = 0, the policy should always exploit the best action."

def test_explore_probability(policy, mocker):
    """Test the probability of exploration."""
    policy.set_epsilon(1.0)  # Always explore
    mocker.patch("numpy.random.choice", return_value=1)  # Mock random choice to return index 1
    q_values = torch.tensor([1.0, 2.0, 3.0])
    action = policy.select_action(q_values)
    assert action == 1, "With epsilon = 1, the policy should always explore and select random actions."
