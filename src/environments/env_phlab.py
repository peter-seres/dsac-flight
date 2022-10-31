from typing import Optional
import gym
import numpy as np
import copy
from environments.base_env import BaseEnv
from src.environments.citation_wrapper import CitationDynamics
from signals import Signal, Const, RandomizedCosineStepSequence


class PhlabEnv(CitationDynamics, BaseEnv):
    """
    Gym environment wrapper that implements the MDP to interact with citation dynamics.

    Observations [7]
    |       0        |        1        |        2      |  3  |   4  |   5  |  6    |
    | c_0 * beta_err | c_1 * theta_err | c_2 * phi_err |  p  |   q  |   r  | alpha |

    Actions [3]
    | 0   | 1    |   2  |
    | del | dail | drud |

    Reward function:
    | norm_1 of clip(c * e, -1, 1) * (-1) |

    """

    def __init__(self, config: dict):
        super().__init__()

        # Save the config
        self.config: dict = config

        # Maximum episode time
        self.t_max: float = config["t_max"]

        # Reference signals (initialized during reset())
        self.beta_ref: Optional[Signal] = None
        self.theta_ref: Optional[Signal] = None
        self.phi_ref: Optional[Signal] = None

        # Reward signal scalar weight
        c = config["c"]
        self.c = np.array(c) * 6 / np.pi

    def set_references(
        self, theta_ref: Signal, phi_ref: Signal, beta_ref: Signal
    ) -> None:
        """Set theta, phi and beta reference signals."""
        self.theta_ref = theta_ref
        self.phi_ref = phi_ref
        self.beta_ref = beta_ref

    @property
    def action_space(self) -> gym.spaces.Box:
        return gym.spaces.Box(
            low=self.defl_low,
            high=self.defl_high,
        )

    @property
    def observation_space(self) -> gym.spaces.Box:
        return gym.spaces.Box(
            low=np.array([-100.0] * 7),
            high=np.array([100.0] * 7),
        )

    def get_observation(self) -> np.ndarray:
        """
        Return observation vector (9):
            [weighted tracking error (3), control surface deflections (3), angular velocities (3) ]
        """
        # K.Dally's version multiplies the error with c
        error = self.c * self.get_error()
        omega = self.x[0:3]
        return np.array([*error, *omega, self.alpha])

    def get_reference(self) -> np.ndarray:
        """Fetch the reference signal values at current time-step t."""
        beta_ref = self.beta_ref(self.t)
        theta_ref = self.theta_ref(self.t)
        phi_ref = self.phi_ref(self.t)

        return np.array([beta_ref, theta_ref, phi_ref])

    def get_controlled_state(self) -> np.ndarray:
        """Fetch the controlled states: [beta, theta, phi]"""
        return np.array([self.beta, self.theta, self.phi])

    def get_error(self) -> np.ndarray:
        """State error: [beta_ref - beta, theta_ref - theta, phi_ref - phi]"""
        return self.get_reference() - self.get_controlled_state()

    def get_reward(self) -> float:
        """Clipped reward function with norm-1."""
        error = self.c * self.get_error()
        error = np.clip(error, -1, 1)
        return -(1 / 3) * np.linalg.norm(error, ord=1)

    def generate_training_signals(self) -> (Signal, Signal, Signal):
        # Pitch angle sequence
        pitch_config = self.config["training"]["pitch_ref"]
        pitch_ref = RandomizedCosineStepSequence(
            t_max=self.t_max,
            block_width=pitch_config["block_width"],
            smooth_width=pitch_config["smooth_width"],
            vary_timings=pitch_config["vary_timings"],
            ampl_max=np.deg2rad(pitch_config["max_ampl"]),
            n_levels=pitch_config["nr_levels"],
            start_with_zero=True,
        )

        # Roll angle sequence
        roll_config = self.config["training"]["roll_ref"]
        roll_ref = RandomizedCosineStepSequence(
            t_max=self.t_max,
            block_width=roll_config["block_width"],
            smooth_width=roll_config["smooth_width"],
            vary_timings=roll_config["vary_timings"],
            ampl_max=np.deg2rad(roll_config["max_ampl"]),
            n_levels=roll_config["nr_levels"],
            start_with_zero=True,
        )

        # Sideslip angle referene is always zero
        beta_ref = Const(value=0.0)

        return pitch_ref, roll_ref, beta_ref

    def get_info(self) -> dict:
        return {
            "t": self.t,
            "x": copy.deepcopy(self.x),
            "u": copy.deepcopy(self.u),
            "ref": self.get_reference(),
        }

    def reset(self, **kwargs) -> (np.ndarray, dict):
        super().initialize()

        # If the references are not set, set them to random sequences:
        if not all([self.theta_ref, self.phi_ref, self.beta_ref]):
            (
                self.theta_ref,
                self.phi_ref,
                self.beta_ref,
            ) = self.generate_training_signals()

        # Return the state and the info
        return self.get_observation(), self.get_info()

    def step(self, action: np.ndarray) -> (np.ndarray, float, bool, dict):
        """Gym-like step function returns: (state, reward, done, info)"""

        # Clip the action vector between [-1, 1]
        clipped_action = action.clip(-1.0, 1.0)

        # Scale [1, 1] action to the incremental control action space
        u = self.scale_action(action=clipped_action)

        # Step the citation dynamics
        super().step(u=u)

        # Return (obs, reward, done, info)
        observation = self.get_observation()
        reward = self.get_reward()
        is_done = self.t >= self.t_max
        info = self.get_info()

        # Handle flight deck
        if self.h <= 100.0:
            reward += (self.t_max - self.t) * self.config["training"]["loc_penalty"]
            is_done = True

        return observation, reward, is_done, info
