import pytest
import numpy as np
from gymnasium.envs.toy_text.frozen_lake import generate_random_map
from deeprl.classic.value_iteration_agent import ValueIterationAgent
from deeprl.environments import GymnasiumEnvWrapper

@pytest.fixture
def frozen_lake_env():
    """Fixture for initializing a non-slippery FrozenLake environment."""
    random_map = generate_random_map(size=4)
    return GymnasiumEnvWrapper('FrozenLake-v1', desc=random_map, is_slippery=False)

def test_value_iteration_convergence(frozen_lake_env):
    """Test that the value iteration algorithm converges."""
    agent = ValueIterationAgent(frozen_lake_env)
    agent.value_iteration()
    
    # Check that the final value function is non-trivial (indicating learning)
    assert np.any(agent.V > 0), "Value function should have non-zero entries after learning."

def test_act_returns_valid_action(frozen_lake_env):
    """Test that the act method returns a valid action."""
    agent = ValueIterationAgent(frozen_lake_env)
    agent.learn()  # Perform value iteration to set up policy
    
    state = frozen_lake_env.reset()
    action = agent.act(state)
    assert 0 <= action < frozen_lake_env.action_space.n, "Action should be within the valid action space."

def test_save_and_load(frozen_lake_env, tmp_path):
    """Test saving and loading the agent's parameters."""
    agent = ValueIterationAgent(frozen_lake_env)
    agent.learn()
    
    filepath = tmp_path / "agent_params.json"
    agent.save(filepath)
    
    # Create a new agent and load saved parameters
    new_agent = ValueIterationAgent(frozen_lake_env)
    new_agent.load(filepath)
    
    # Ensure the value table and policy match after loading
    assert np.array_equal(agent.V, new_agent.V), "Loaded value table does not match the saved table."
    assert np.array_equal(agent.policy.policy_table, new_agent.policy.policy_table), "Loaded policy does not match the saved policy."

