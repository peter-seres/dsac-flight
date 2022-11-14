"""
This module defnines a simple a replay buffer that stores transitions in the MDP.
"""

import torch
import numpy as np
import random
import functools
from collections import namedtuple, deque

# Transition and batch are all tuples of <s, a, r, s', done> data:
Transition = namedtuple("Transition", "s a r ns d")
Batch = namedtuple("Batch", "s a r ns d")


class ReplayBuffer(object):
    def __init__(self, buffer_size: int, device: torch.device):
        self.device = device
        self.buffer_size = buffer_size
        self.memory = deque(maxlen=buffer_size)

        # Data type and device management
        self.torchify = functools.partial(
            torch.tensor, dtype=torch.float, device=self.device
        )

    def push(self, transition: Transition) -> None:
        self.memory.appendleft(transition)

    def ready_for(self, batch_size: int) -> bool:
        if len(self.memory) >= batch_size:
            return True
        return False

    def sample(self, batch_size: int) -> Batch:
        """Sample a batch of transition uniformly from the buffer."""

        # Random sampling
        batch = random.sample(self.memory, batch_size)
        batch = Batch(*zip(*batch))

        # Convert to torch arrays
        s = self.torchify(np.array(batch.s)).view(batch_size, -1)
        a = self.torchify(np.array(batch.a)).view(batch_size, -1)
        r = self.torchify(np.array(batch.r)).view(batch_size, 1)
        ns = self.torchify(np.array(batch.ns)).view(batch_size, -1)
        d = self.torchify(np.array(batch.d)).view(batch_size, 1)

        return Batch(s, a, r, ns, d)
