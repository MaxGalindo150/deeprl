import pytest
import numpy as np
from deeprl.reward_shaping.potential_based_shaping import PotentialBasedShaping

@pytest.fixture
def potential_function():
    """Fixture to provide a simple potential function based on the distance to the goal."""
    goal_state = np.array([1.0, 1.0])  # Example goal state
    return lambda state: -np.linalg.norm(np.array(state) - goal_state)

@pytest.fixture
def reward_shaping(potential_function):
    """Fixture to create an instance of PotentialBasedShaping."""
    return PotentialBasedShaping(potential_function=potential_function, discount_factor=0.99)

def test_reward_shaping_increases_with_progress(reward_shaping):
    """
    Test that the reward increases when the agent moves closer to the goal.
    """
    state = [0.0, 0.0]  # Initial state
    next_state = [0.5, 0.5]  # Closer to the goal
    reward = -1  # Original environment reward

    shaped_reward = reward_shaping.shape(state, None, next_state, reward)
    assert shaped_reward > reward, "Shaped reward should increase as the agent progresses towards the goal."

def test_reward_shaping_decreases_with_regression(reward_shaping):
    """
    Test that the reward decreases when the agent moves away from the goal.
    """
    state = [0.5, 0.5]  # Starting closer to the goal
    next_state = [0.0, 0.0]  # Moves away from the goal
    reward = -1  # Original environment reward

    shaped_reward = reward_shaping.shape(state, None, next_state, reward)
    assert shaped_reward < reward, "Shaped reward should decrease as the agent regresses away from the goal."

def test_reward_shaping_no_change_at_goal(reward_shaping):
    """
    Test that the reward does not change if the agent remains at the goal.
    """
    goal_state = [1.0, 1.0]  # Goal position
    state = goal_state
    next_state = goal_state  # Remains at the goal
    reward = -1  # Original environment reward

    shaped_reward = reward_shaping.shape(state, None, next_state, reward)
    assert shaped_reward == pytest.approx(reward), "Shaped reward should remain unchanged when staying at the goal."

def test_shaping_term_correctness(reward_shaping, potential_function):
    """
    Test that the shaping term is computed correctly.
    """
    state = [0.0, 0.0]
    next_state = [0.5, 0.5]
    reward = -1

    potential_current = potential_function(state)
    potential_next = potential_function(next_state)
    expected_shaping_term = reward_shaping.discount_factor * potential_next - potential_current
    expected_reward = reward + expected_shaping_term

    shaped_reward = reward_shaping.shape(state, None, next_state, reward)
    assert shaped_reward == pytest.approx(expected_reward), "Shaped reward does not match the expected value."
