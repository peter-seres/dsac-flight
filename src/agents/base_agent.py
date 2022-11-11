"""
This module defines an abstract class for RL agents.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any
import numpy as np
import torch


class BaseAgent(ABC):
    """Interface for Agents."""

    def __init__(
        self, device: torch.device, config: dict, obs_dim: int, action_dim: int
    ):
        self.device = device
        self.config = config
        self.obs_dim = obs_dim
        self.action_dim = action_dim

    @abstractmethod
    def act(self, state: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def act_greedy(self, state: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def every_sample_update_callback(self) -> None:
        """Callback after each transition"""
        raise NotImplementedError

    @abstractmethod
    def update(self, state, action, reward, next_state, done) -> Any:
        """Update the networks of the agent."""
        raise NotImplementedError

    @abstractmethod
    def get_eta(self) -> float:
        """Get the temperature coefficient."""
        # todo: Move to soft actor-critic class, as this is SAC specific
        raise NotImplementedError

    @abstractmethod
    def set_eval(self):
        """Set the agent to eval mode."""
        raise NotImplementedError

    @abstractmethod
    def save_to_file(self, file_path: str) -> None:
        """Implement how to save the agent. Usually .pth for the NN parameters."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def load_from_file(
        file_path: str,
        config: dict,
        device: torch.device,
        obs_dim: int,
        action_dim: int,
    ) -> BaseAgent:
        """Implement how to load the agent. Usually from .pth."""
        pass
