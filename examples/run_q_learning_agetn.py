from deeprl.environments import GymnasiumEnvWrapper
from deeprl.agents.q_learning_agent import QLearningAgent
from deeprl.policies.epsilon_greedy_decay_policy import EpsilonGreedyDecayPolicy
from deeprl.reward_shaping.step_penalty_shaping import StepPenaltyShaping

def main():
    
    # Configure the FrozenLake environment
    env = GymnasiumEnvWrapper('FrozenLake-v1', is_slippery=False)
    
    # Initialize the agent with a decaying epsilon-greedy policy
    policy = EpsilonGreedyDecayPolicy(epsilon=1, decay_rate=0.99, min_epsilon=0.1)
    
    agent = QLearningAgent(
        env=env, 
        policy=policy,
        reward_shaping=StepPenaltyShaping(step_penalty=-0.1),
        verbose=True
    )
    
    # Train the agent
    agent.learn(episodes=100000, max_steps=10000, save_train_graph=True)
    
    # Evaluate the agent
    rewards = agent.interact(episodes=10, render=True, save_test_graph=True)

if __name__ == '__main__':
    main()