import pytest
import numpy as np
from gymnasium.envs.toy_text.frozen_lake import generate_random_map
from deeprl.classic.policy_iteration_agent import PolicyIterationAgent
from deeprl.environments import GymnasiumEnvWrapper

@pytest.fixture
def frozen_lake_env():
    """Fixture for initializing a non-slippery FrozenLake environment."""
    random_map = generate_random_map(size=4)
    return GymnasiumEnvWrapper('FrozenLake-v1', desc=random_map, is_slippery=False)

def test_policy_evaluation_convergence(frozen_lake_env):
    """Test that the policy evaluation algorithm converges."""
    agent = PolicyIterationAgent(frozen_lake_env)
    agent.policy_iteration()
    
    # Check that the value function is non-trivial (indicating learning)
    assert np.any(agent.value_table > 0), "Value table should have non-zero entries after policy evaluation."

def test_policy_stability(frozen_lake_env):
    """Test that the policy becomes stable after convergence."""
    agent = PolicyIterationAgent(frozen_lake_env)
    agent.policy_iteration()
    
    # Check that the policy is stable (no actions should change)
    policy_stable = agent.update_policy()
    assert policy_stable, "Policy should be stable after full policy iteration."

def test_act_returns_valid_action(frozen_lake_env):
    """Test that the act method returns a valid action."""
    agent = PolicyIterationAgent(frozen_lake_env)
    agent.learn()  # Perform policy iteration to set up policy
    
    state = frozen_lake_env.reset()
    action = agent.act(state)
    assert 0 <= action < frozen_lake_env.action_space.n, "Action should be within the valid action space."



def test_save_and_load(frozen_lake_env, tmp_path):
    """Test saving and loading the agent's parameters."""
    agent = PolicyIterationAgent(frozen_lake_env)
    agent.learn()
    
    filepath = tmp_path / "agent_params.json"
    agent.save(filepath)
    
    # Create a new agent and load saved parameters
    new_agent = PolicyIterationAgent(frozen_lake_env)
    new_agent.load(filepath)
    
    # Ensure the value table and policy match after loading
    assert np.array_equal(agent.value_table, new_agent.value_table), "Loaded value table does not match the saved table."
    assert np.array_equal(agent.policy.policy_table, new_agent.policy.policy_table), "Loaded policy does not match the saved policy."
