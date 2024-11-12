from deeprl.environments import GymnasiumEnvWrapper
from deeprl.agents.q_learning_agent import QLearningAgent
from deeprl.policies import EpsilonGreedyPolicy
from deeprl.reward_shaping.step_penalty_shaping import StepPenaltyShaping
from deeprl.common.evaluate_policy import evaluate_policy

def main():
    
    # Configure the FrozenLake environment
    env = GymnasiumEnvWrapper('FrozenLake-v1', is_slippery=False)
    
    # Initialize the agent with a decaying epsilon-greedy policy
    policy = EpsilonGreedyPolicy(epsilon=0.1)
    
    agent = QLearningAgent(
        env=env, 
        policy=policy,
        reward_shaping=StepPenaltyShaping(step_penalty=-0.1),
        verbose=True
    )
    
    # Train the agent
    agent.learn(episodes=100000, max_steps=10000)
    
    # Evaluate the agent
    mean_reward, std_reward = evaluate_policy(agent, agent.get_env(), num_eval_episodes=10)
    
    #agent.save('frozenlake_q_learning_agent')
    #agent.print_saved_config('frozenlake_q_learning_agent')
    print(f'Mean reward: {mean_reward}, Std reward: {std_reward}')
if __name__ == '__main__':
    main()