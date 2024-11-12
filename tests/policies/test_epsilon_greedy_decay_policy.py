import pytest
import torch
import numpy as np
from deeprl.policies.epsilon_greedy_decay_policy import EpsilonGreedyDecayPolicy

@pytest.fixture
def policy():
    """Fixture to create an EpsilonGreedyDecayPolicy instance."""
    return EpsilonGreedyDecayPolicy(epsilon=0.5, decay_rate=0.9, min_epsilon=0.1)

def test_policy_initialization(policy):
    """Test that the policy is initialized with correct values."""
    assert policy.epsilon == 0.5, "Epsilon should initialize to 0.5."
    assert policy.decay_rate == 0.9, "Decay rate should initialize to 0.9."
    assert policy.min_epsilon == 0.1, "Minimum epsilon should initialize to 0.1."

def test_select_action_exploit(policy, monkeypatch):
    """Test that the policy selects the action with the highest Q-value (exploit)."""
    monkeypatch.setattr(torch, "rand", lambda _: torch.tensor([0.6]))  # Ensure epsilon condition is not met
    q_values = torch.tensor([1.0, 2.0, 3.0])  # Highest Q-value at index 2
    action = policy.select_action(q_values)
    assert action == 2, "The policy should select the action with the highest Q-value."

def test_select_action_explore(policy, monkeypatch):
    """Test that the policy selects a random action (explore)."""
    monkeypatch.setattr(torch, "rand", lambda _: torch.tensor([0.3]))  # Ensure epsilon condition is met
    q_values = torch.tensor([1.0, 2.0, 3.0])
    random_action = policy.select_action(q_values)
    assert random_action in range(len(q_values)), "The action should be a valid random choice."

def test_epsilon_decay(policy):
    """Test that epsilon decays correctly and respects the minimum epsilon."""
    initial_epsilon = policy.epsilon
    policy.update()
    assert policy.epsilon == max(policy.min_epsilon, initial_epsilon * policy.decay_rate), \
        "Epsilon should decay correctly and not fall below the minimum."

def test_min_epsilon(policy):
    """Test that epsilon does not decay below the minimum value."""
    policy.epsilon = 0.1  # Set epsilon to the minimum value
    policy.update()
    assert policy.epsilon == 0.1, "Epsilon should not decay below the minimum value."

def test_set_epsilon(policy):
    """Test that epsilon can be updated manually."""
    policy.set_epsilon(0.8)
    assert policy.epsilon == 0.8, "Epsilon should update to the new value."
