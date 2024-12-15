import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import torch as t
import torch.nn as nn
from collections import deque
import random
from DQN import DQN
from ReplayMemory import ReplayMemory
from ForzenLakeDQL import *

if __name__ == '__main__':
      frozen_lake= FrozenLakeDQL()
      is_slippery= False
      frozen_lake.train(1000,is_slippery=is_slippery)
      frozen_lake.test(4,is_slippery=is_slippery)


