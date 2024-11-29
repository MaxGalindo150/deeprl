from deeprl.environments import GymnasiumEnvWrapper
from deeprl.classic.q_learning_agent import QLearningAgent
from deeprl.function_approximations import RBFBasisApproximator
from deeprl.reward_shaping import MountainCarRewardShaping
from deeprl.common.evaluate_policy import evaluate_policy

def main():
    
    # Initialize the environment and approximator
    env = GymnasiumEnvWrapper('MountainCar-v0')
    approximator = RBFBasisApproximator(env=env, gamma=0.5, n_components=500)
        
    agent = QLearningAgent(
        env=env,
        learning_rate=0.1,
        discount_factor=0.99,
        is_continuous=True,
        approximator=approximator,
        reward_shaping=MountainCarRewardShaping(),
        verbose=True
    )
    
    # Train the agent
    agent.learn(episodes=10000, max_steps=10000)
    
    # Evaluate the agent
    mean_reward, std_reward = evaluate_policy(agent, agent.get_env(), num_eval_episodes=10)
if __name__ == '__main__':
    main()