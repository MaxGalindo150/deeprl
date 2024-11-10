from deeprl.environments import GymnasiumEnvWrapper
from deeprl.agents.q_learning_agent import QLearningAgent
from deeprl.function_approximations import RBFBasisApproximator
from deeprl.reward_shaping import MountainCarRewardShaping

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
    agent.learn(episodes=10000, max_steps=10000, save_train_graph=True)
    
    # Evaluate the agent
    rewards = agent.interact(episodes=10, render=True, save_test_graph=True)

if __name__ == '__main__':
    main()