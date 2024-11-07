import pytest
import torch
from deeprl.policies.deterministic_policy import DeterministicPolicy

@pytest.fixture
def policy():
    """Fixture to create a DeterministicPolicy instance with a mock observation space of 10 states."""
    class MockObservationSpace:
        n = 10  # Define the number of states for the observation space
    return DeterministicPolicy(observation_space=MockObservationSpace())

def test_policy_initialization(policy):
    """Test that the policy table is initialized correctly."""
    assert len(policy.policy_table) == 10, "Policy table should have 10 entries."
    assert torch.equal(policy.policy_table, torch.zeros(10)), "Policy table should initialize with all zeros."

def test_select_action_valid_state(policy):
    """Test selecting an action for a valid state."""
    # Set a known action for a specific state
    policy.update_policy(3, 2)
    assert policy.select_action(3) == 2, "select_action should return the updated action for the given state."

def test_select_action_invalid_state(policy):
    """Test selecting an action for an invalid state index."""
    with pytest.raises(IndexError, match="State 15 is out of range for the policy table."):
        policy.select_action(15)  # State 15 is out of range for a 10-state policy table

def test_update_policy(policy):
    """Test updating the policy for a given state."""
    policy.update_policy(5, 3)  # Set action 3 for state 5
    assert policy.policy_table[5] == 3, "Policy table entry for state 5 should be updated to 3."

def test_update_policy_invalid_state(policy):
    """Test that an invalid state index raises an IndexError during update."""
    with pytest.raises(IndexError, match="State 15 is out of range for the policy table."):
        policy.update_policy(15, 1)  # Attempt to update an out-of-bounds state index

def test_update_policy_overwrite(policy):
    """Test updating an existing state with a new action value."""
    policy.update_policy(4, 2)  # Set action 2 for state 4
    assert policy.select_action(4) == 2, "Initial action for state 4 should be 2."
    
    policy.update_policy(4, 5)  # Update action for state 4 to 5
    assert policy.select_action(4) == 5, "Action for state 4 should update to 5."
