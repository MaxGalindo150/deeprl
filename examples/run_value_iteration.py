from deeprl.environments import GymnasiumEnvWrapper
from deeprl.agents import ValueIterationAgent

def main():
    # Configure the FrozenLake environment with a slippery surface using GymnasiumEnvWrapper from DeepRL
    env = GymnasiumEnvWrapper('FrozenLake-v1', render_mode='human')
    agent = ValueIterationAgent(env)
    agent.learn()
    
    agent.save('value_iteration_agent.json')

    # Unpack the initial state and reset the environment
    agent.interact(num_episodes=1, render=True)

if __name__ == '__main__':
    main()
