from deeprl.agents import QLearningAgent
from deeprl.environments import GymnasiumEnvWrapper

def main():
    env = GymnasiumEnvWrapper('FrozenLake-v1', is_slippery=False)
    
    agent = QLearningAgent(
        env=env, 
        learning_rate=0.1, 
        discount_factor=0.99, 
        epsilon=0.1, 
        step_penalty=-0.1,
        verbose=True)
    
    agent.learn(episodes=100000, max_steps=1000)
    
    episodes = 1
    agent.interact(episodes=episodes, max_steps=100, render=True)
     
    agent.save('q_learning_agent.json')

if __name__ == '__main__':
    main()