from typing import Optional
import gym
import numpy as np
from abc import ABC, abstractmethod

gym.logger.set_level(40)


class BaseEnv(gym.Env, ABC):
    """Base class to write gym-like environments for control."""

    @abstractmethod
    def set_references(self, *args, **kwargs):
        """Implement to allow custom evaluation routines."""
        raise NotImplementedError

    @property
    @abstractmethod
    def action_space(self) -> gym.spaces.Box:
        raise NotImplementedError

    @property
    @abstractmethod
    def observation_space(self) -> gym.spaces.Box:
        raise NotImplementedError

    @property
    def action_dim(self) -> int:
        return self.action_space.shape[0]

    @property
    def obs_dim(self) -> int:
        return self.observation_space.shape[0]

    def scale_action(self, action: np.ndarray) -> np.ndarray:
        """Scale the actions from [-1, 1] to the appropriate scale of the action space."""
        low, high = self.action_space.low, self.action_space.high
        action = low + 0.5 * (action + 1.0) * (high - low)
        return action

    def render(self, mode="human"):
        """Implement the render method so that derived environments don't have to."""
        pass

    @abstractmethod
    def reset(
        self,
        *,
        is_eval: bool = False,
        seed: Optional[int] = None,
        return_info: bool = False,
        options: Optional[dict] = None,
    ) -> (np.ndarray, dict):
        raise NotImplementedError
