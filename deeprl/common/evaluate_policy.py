import numpy as np

def evaluate_policy(model, env, num_eval_episodes=10):
    """
    Evaluates the given policy model over a specified number of episodes.

    :param model: The policy model to evaluate. This should have an `act` method for action selection.
    :param env: The environment to evaluate the policy in.
    :param num_eval_episodes: The number of episodes to evaluate the policy over.
    :return: A tuple (mean_reward, std_reward), where:
        - mean_reward: The average reward obtained over all episodes.
        - std_reward: The standard deviation of the rewards obtained over all episodes.
    """
    rewards = []
    print("Evaluating policy...")

    for episode in range(num_eval_episodes):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            action = model.act(state)  # Get action from the model
            next_state, reward, done, truncated, _ = env.step(action)
            total_reward += reward
            state = next_state

            if done or truncated:
                break

        rewards.append(total_reward)

    mean_reward = np.mean(rewards)
    std_reward = np.std(rewards)

    return mean_reward, std_reward
