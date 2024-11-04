import unittest
import gymnasium
from deeprl.agents import ValueIterationAgent

class TestValueIterationAgent(unittest.TestCase):
    def test_value_iteration_agent(self):
        env = gymnasium.make('FrozenLake-v1', is_slippery=False)
        agent = ValueIterationAgent(env)
        agent.learn()
        # Verifica que la política derivada sea la óptima esperada
        expected_policy = [RIGHT, DOWN, DOWN, DOWN,
                           RIGHT, RIGHT, DOWN, DOWN,
                           RIGHT, RIGHT, RIGHT, DOWN,
                           RIGHT, RIGHT, RIGHT, TERMINAL_STATE]
        self.assertEqual(list(agent.policy), expected_policy)

if __name__ == '__main__':
    unittest.main()
