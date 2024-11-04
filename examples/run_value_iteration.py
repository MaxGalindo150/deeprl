import gymnasium
from deeprl.agents import ValueIterationAgent

def main():
    # Configure the FrozenLake environment with a slippery surface
    env = gymnasium.make('FrozenLake-v1', is_slippery=True, render_mode='human')
    agent = ValueIterationAgent(env)
    agent.learn()

    # Unpack the initial state and reset the environment
    state, _ = env.reset()
    done = False
    env.render()  # Render the initial state

    while not done:
        action = agent.act(state)
        state, reward, done, truncated, info = env.step(action)
        env.render()  # Render the updated state

    # Close the environment
    env.close()

if __name__ == '__main__':
    main()
