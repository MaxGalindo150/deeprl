from deeprl.agents import PolicyIterationAgent
from deeprl.environments import GymnasiumEnvWrapper

def main():
    # Configure the FrozenLake environment with a slippery surface using GymnasiumEnvWrapper from DeepRL
    env = GymnasiumEnvWrapper('FrozenLake-v1',is_slippery=True, render_mode='human')
    agent = PolicyIterationAgent(env)
    agent.learn()

    # Unpack the initial state and reset the environment
    agent.interact(num_episodes=1, render=True)

if __name__ == '__main__':
    main()