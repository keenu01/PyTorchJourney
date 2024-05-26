import torch as t
import torch.nn as nn
from collections import deque
import random
class DQN(nn.Module):
      def __init__(self,IN:int,HID:int,OUT:int):
            super().__init__()
            self.block1 = nn.Sequential(
                  
                  nn.Linear(IN,HID),
                  nn.Linear(HID,HID),
                  nn.Linear(HID,HID),
                  nn.Linear(HID,HID),
                  nn.Linear(HID,HID),
                  nn.Linear(HID,OUT)
            )
      
      def forward(self,x:t.Tensor)-> t.Tensor:
           return self.block1(x)
