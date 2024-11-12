from deeprl.agents import PolicyIterationAgent
from deeprl.environments import GymnasiumEnvWrapper
from deeprl.common.evaluate_policy import evaluate_policy

def main():
    # Configure the FrozenLake environment with a slippery surface using GymnasiumEnvWrapper from DeepRL
    env = GymnasiumEnvWrapper('FrozenLake-v1',is_slippery=False)
    agent = PolicyIterationAgent(env)
    agent.learn()
    
    #agent.save('policy_iteration_agent.json')

    # Unpack the initial state and reset the environment
    mean_reward, std_reward = evaluate_policy(agent, agent.get_env(), num_eval_episodes=100)
    
    print(f'Mean reward: {mean_reward}, Std reward: {std_reward}')

if __name__ == '__main__':
    main()