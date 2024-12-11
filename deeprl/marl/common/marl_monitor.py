import pettingzoo
import time

from deeprl.common.monitor import ResultsWriter

from typing import Union, Tuple

class MARLMonitor:
    """
    A monitor wrapper for PettingZoo environments, used to log episodes, rewards, lengths, times, and other data.
    
    :param env: The PettingZoo environment (AECEnv or ParallelEnv)
    :param filename: The location to save a log file, can be None for no log
    :param allow_early_resets: Allows resetting the environment before it is done
    :param reset_keywords: Extra keywords for the reset call, if extra parameters are needed at reset
    :param info_keywords: Extra information to log, from the information return of env.step()
    :param override_existing: If True, overrides existing files; otherwise, appends to the file.
    """
    EXT = "marl_monitor.csv"
    
    def __init__(
        self,
        env: Union[pettingzoo.AECEnv, pettingzoo.ParallelEnv],
        filename: str = None,
        allow_early_resets: bool = True,
        reset_keywords: Tuple[str, ...] = (),
        info_keywords: Tuple[str, ...] = (),
        override_existing: bool = True,
    ):
        self.env = env
        self.t_start = time.time()
        self.allow_early_resets = allow_early_resets
        self.reset_keywords = reset_keywords
        self.info_keywords = info_keywords
        self.episode_rewards = {agent: 0 for agent in env.possible_agents}
        self.episode_lengths = {agent: 0 for agent in env.possible_agents}
        self.results_writer = None

        if filename is not None:
            env_id = env.metadata["name"] if env.metadata else "unknown_env"
            self.results_writer = ResultsWriter(
                filename,
                header={"t_start": self.t_start, "env_id": str(env_id)},
                extra_keys=("agent_id", "reward", "length") + reset_keywords + info_keywords,
                override_existing=override_existing,
            )
    
    def reset(self, **kwargs):
        self.episode_rewards = {agent: 0 for agent in self.env.possible_agents}
        self.episode_lengths = {agent: 0 for agent in self.env.possible_agents}
        return self.env.reset(**kwargs)
    
    def step(self, actions):
        obs, rewards, terminations, truncations, infos = self.env.step(actions)

        for agent in self.env.agents:
            if agent in rewards:
                self.episode_rewards[agent] += rewards[agent]
                self.episode_lengths[agent] += 1

            if terminations.get(agent, False) or truncations.get(agent, False):
                ep_info = {
                    "agent_id": agent,
                    "reward": round(self.episode_rewards[agent], 6),
                    "length": self.episode_lengths[agent],
                }
                print(f"Episode info: {ep_info}")
                if self.results_writer:
                    self.results_writer.write_row(ep_info)
                
                # Reset agent-specific counters
                self.episode_rewards[agent] = 0
                self.episode_lengths[agent] = 0

        return obs, rewards, terminations, truncations, infos

    def close(self):
        if self.results_writer:
            self.results_writer.close()
        self.env.close()