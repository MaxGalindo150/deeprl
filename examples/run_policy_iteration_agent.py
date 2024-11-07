from deeprl.agents import PolicyIterationAgent
from deeprl.environments import GymnasiumEnvWrapper

def main():
    # Configure the FrozenLake environment with a slippery surface using GymnasiumEnvWrapper from DeepRL
    env = GymnasiumEnvWrapper('FrozenLake-v1',is_slippery=False, render_mode='human')
    agent = PolicyIterationAgent(env)
    agent.learn()
    
    #agent.save('policy_iteration_agent.json')

    # Unpack the initial state and reset the environment
    agent.interact(num_episodes=3, render=True)

if __name__ == '__main__':
    main()