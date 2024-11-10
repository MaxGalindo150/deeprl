import pytest
import numpy as np
from deeprl.reward_shaping.distance_based_shaping import DistanceBasedShaping

@pytest.fixture
def shaping():
    """Fixture to create a DistanceBasedShaping instance with a goal state."""
    goal_state = [1.0, 1.0]
    return DistanceBasedShaping(goal_state)

def test_shape_reward_increase_with_closer_state(shaping):
    """
    Test that the reward increases as the next state gets closer to the goal state.
    """
    state = np.array([0.0, 0.0])
    next_state = np.array([0.5, 0.5])  # Closer to the goal
    action = 1
    reward = 10

    shaped_reward = shaping.shape(state, action, next_state, reward)
    expected_shaped_reward = reward - np.linalg.norm(next_state - np.array([1.0, 1.0]))

    assert shaped_reward == pytest.approx(expected_shaped_reward), \
        "Shaped reward should correctly account for the distance to the goal."

def test_shape_reward_with_zero_distance(shaping):
    """
    Test that the reward remains unchanged when the next state equals the goal state.
    """
    state = np.array([0.0, 0.0])
    next_state = np.array([1.0, 1.0])  # Exactly at the goal
    action = 0
    reward = 10

    shaped_reward = shaping.shape(state, action, next_state, reward)
    expected_shaped_reward = reward - np.linalg.norm(next_state - np.array([1.0, 1.0]))

    assert shaped_reward == pytest.approx(expected_shaped_reward), \
        "Shaped reward should be equal to the original reward when at the goal."

def test_shape_reward_with_farther_state(shaping):
    """
    Test that the reward decreases as the next state gets farther from the goal state.
    """
    state = np.array([0.0, 0.0])
    next_state = np.array([2.0, 2.0])  # Farther from the goal
    action = 2
    reward = 10

    shaped_reward = shaping.shape(state, action, next_state, reward)
    expected_shaped_reward = reward - np.linalg.norm(next_state - np.array([1.0, 1.0]))

    assert shaped_reward == pytest.approx(expected_shaped_reward), \
        "Shaped reward should correctly decrease as distance to the goal increases."

def test_shape_with_negative_original_reward(shaping):
    """
    Test that the reward shaping works correctly with a negative original reward.
    """
    state = np.array([0.0, 0.0])
    next_state = np.array([0.5, 0.5])
    action = 1
    reward = -5

    shaped_reward = shaping.shape(state, action, next_state, reward)
    expected_shaped_reward = reward - np.linalg.norm(next_state - np.array([1.0, 1.0]))

    assert shaped_reward == pytest.approx(expected_shaped_reward), \
        "Shaped reward should correctly handle negative original rewards."

def test_shape_with_goal_state_far_away(shaping):
    """
    Test that the shaped reward accounts for large distances to a distant goal.
    """
    shaping.goal_state = np.array([10.0, 10.0])
    state = np.array([0.0, 0.0])
    next_state = np.array([1.0, 1.0])
    action = 1
    reward = 0

    shaped_reward = shaping.shape(state, action, next_state, reward)
    expected_shaped_reward = reward - np.linalg.norm(next_state - np.array([10.0, 10.0]))

    assert shaped_reward == pytest.approx(expected_shaped_reward), \
        "Shaped reward should account for large distances to the goal."
