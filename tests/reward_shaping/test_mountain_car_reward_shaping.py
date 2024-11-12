import pytest
from deeprl.reward_shaping.mountain_car_reward_shaping import MountainCarRewardShaping

@pytest.fixture
def reward_shaping():
    """Fixture to create a MountainCarRewardShaping instance."""
    return MountainCarRewardShaping()

def test_reward_shaping_progress(reward_shaping):
    """
    Test that the reward increases when the car progresses towards the goal.
    """
    state = (-0.5, 0.0)  # Initial position and velocity
    next_state = (-0.4, 0.1)  # Next position and velocity
    reward = -1  # Original reward from the environment

    shaped_reward = reward_shaping.shape(state, None, next_state, reward)
    expected_reward = reward + 10 * (next_state[0] - state[0])

    assert shaped_reward == pytest.approx(expected_reward), \
        "Shaped reward should increase with progress towards the goal."

def test_reward_shaping_goal_reached(reward_shaping):
    """
    Test that the reward includes a bonus when the goal is reached.
    """
    state = (0.4, 0.1)  # Position and velocity
    next_state = (0.5, 0.0)  # Goal position reached
    reward = -1  # Original reward from the environment

    shaped_reward = reward_shaping.shape(state, None, next_state, reward)
    expected_reward = reward + 10 * (next_state[0] - state[0]) + 100  # Goal bonus

    assert shaped_reward == pytest.approx(expected_reward), \
        "Shaped reward should include a goal bonus when the car reaches the goal."

def test_reward_shaping_no_progress(reward_shaping):
    """
    Test that the reward does not increase when there is no progress.
    """
    state = (-0.5, 0.0)  # Initial position and velocity
    next_state = (-0.5, 0.1)  # No position progress, only velocity change
    reward = -1  # Original reward from the environment

    shaped_reward = reward_shaping.shape(state, None, next_state, reward)
    expected_reward = reward  # No position change, no extra reward

    assert shaped_reward == pytest.approx(expected_reward), \
        "Shaped reward should remain unchanged when there is no position progress."

def test_reward_shaping_negative_progress(reward_shaping):
    """
    Test that the reward decreases when the car moves away from the goal.
    """
    state = (-0.4, 0.0)  # Initial position and velocity
    next_state = (-0.5, -0.1)  # Moves backward
    reward = -1  # Original reward from the environment

    shaped_reward = reward_shaping.shape(state, None, next_state, reward)
    expected_reward = reward + 10 * (next_state[0] - state[0])  # Negative progress

    assert shaped_reward == pytest.approx(expected_reward), \
        "Shaped reward should decrease when the car moves away from the goal."

def test_reward_shaping_stays_at_goal(reward_shaping):
    """
    Test that the reward remains penalized when the car stays at the goal.
    """
    state = (0.5, 0.0)
    next_state = (0.5, 0.0)  
    reward = -100

    shaped_reward = reward_shaping.shape(state, None, next_state, reward)
    expected_reward = 0  

    assert shaped_reward == pytest.approx(expected_reward), \
        "Shaped reward should remain as the time penalty when the car stays at the goal."

