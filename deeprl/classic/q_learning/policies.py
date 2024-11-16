import numpy as np
import torch as th
from gymnasium import spaces
from torch import nn

from typing import Dict, Any, Optional, Type

from deeprl.common.base_policy import BasePolicy
from deeprl.common.type_aliases import PyTorchObs, Schedule
from deeprl.common.torch_layers import (
    BaseFeaturesExtractor,
    FlattenExtractor,
)

class QTable(BasePolicy):
    """
    Action-Value (Q-Value) table for Q-Learning.
    
    :param observation_space: The observation space.
    :param action_space: Action space.
    :param 
    """