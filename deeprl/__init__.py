import os

#from deeprl.a2c import A2C
from deeprl.common.utils import get_system_info
#from deeprl.ddpg import DDPG
from deeprl.deep.dqn import DQN
from deeprl.deep.ppo import PPO
from deeprl.classic.q_learning.q_learning import QLearning
from deeprl.her.her_replay_buffer import HerReplayBuffer
#from deeprl.ppo import PPO
#from deeprl.sac import SAC
#from deeprl.td3 import TD3

# Read version from file
version_file = os.path.join(os.path.dirname(__file__), "version.txt")

with open(version_file) as file_handler:
    __version__ = file_handler.read().strip()


def HER(*args, **kwargs):
    raise ImportError(
        "`HER` is now a replay buffer class `HerReplayBuffer`.\n "
        "Please check the documentation for more information: http://deeprl.sytes.net/"
    )


__all__ = [
    "A2C",
    "DDPG",
    "DQN",
    "PPO",
    "SAC",
    "TD3",
    "QLearning",
    "HerReplayBuffer",
    "get_system_info",
]