import csv
import os
import time
from typing import Optional, List

import gymnasium
from pettingzoo.utils.env import AECEnv
from pettingzoo.utils.wrappers import BaseWrapper
from pettingzoo.butterfly import knights_archers_zombies_v10

class PettingZooMonitor(BaseWrapper):
    """
    Monitor for PettingZoo environments. Logs rewards, episode lengths, and other statistics.
    Outputs to a CSV file for later analysis.

    :param env: The PettingZoo environment to wrap.
    :param filename: Path to the CSV file for logging.
    :param allow_early_resets: Whether to allow logging of early terminations.
    """

    def __init__(self, env: AECEnv, filename: str, allow_early_resets: bool = True):
        super().__init__(env)
        self.t_start = time.time()
        self.allow_early_resets = allow_early_resets
        self.current_episode_rewards = {agent: 0 for agent in env.possible_agents}
        self.current_episode_lengths = {agent: 0 for agent in env.possible_agents}
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_times = []
        self.filename = filename
        self.file_handler = None
        self.csv_writer = None

        if filename is not None:
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            self.file_handler = open(filename, "w", newline="")
            self.csv_writer = csv.writer(self.file_handler)
            self.csv_writer.writerow(["agent", "reward", "length", "time"])

    def step(self, action):
        agent = self.env.agent_selection
        self.current_episode_lengths[agent] += 1

        obs, reward, termination, truncation, info = self.env.step(action)

        # Update rewards
        self.current_episode_rewards[agent] += reward

        if termination or truncation:
            self._log_episode(agent)
            self.current_episode_rewards[agent] = 0
            self.current_episode_lengths[agent] = 0

        return obs, reward, termination, truncation, info

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        self.current_episode_rewards = {agent: 0 for agent in self.env.possible_agents}
        self.current_episode_lengths = {agent: 0 for agent in self.env.possible_agents}
        return self.env.reset(seed=seed, options=options)

    def _log_episode(self, agent):
        episode_time = round(time.time() - self.t_start, 6)
        self.episode_rewards.append(self.current_episode_rewards[agent])
        self.episode_lengths.append(self.current_episode_lengths[agent])
        self.episode_times.append(episode_time)

        if self.csv_writer:
            self.csv_writer.writerow(
                [
                    agent,
                    self.current_episode_rewards[agent],
                    self.current_episode_lengths[agent],
                    episode_time,
                ]
            )

    def close(self):
        if self.file_handler:
            self.file_handler.close()
        super().close()


if __name__ == "__main__":
    env = knights_archers_zombies_v10.parallel_env()
    monitored_env = PettingZooMonitor(env, filename="logs/monitor.csv")
    monitored_env.reset()
    for _ in range(100):
        action = {agent: env.action_space(agent).sample() for agent in env.agents}
        obs, reward, done, info = monitored_env.step(action)
        if done:
            monitored_env.reset()
    monitored_env.close()
# Example usage:
# env = knights_archers_zombies_v10.parallel_env()
# monitored_env = PettingZooMonitor(env, filename="logs/monitor.csv")
