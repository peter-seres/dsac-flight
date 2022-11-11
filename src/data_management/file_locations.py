"""
This module implements a class FileLocationManager that manages file locations
using the convention defined by Naming.
"""
from __future__ import annotations
import os
from dataclasses import dataclass
from data_management.log_utils import make_dir
from typing import Iterator, List


class Naming:
    """Naming convention to save the RL runs into a folder structure."""

    CONFIG = "config.toml"  # hyperparameters
    METADATA = "metadata.toml"  # metadata
    REWARDS = "rewards.npy"  # episodic updates of rewards and adaptive temperature

    class Episodes:
        FOLDER = "episodes"  # Name of the episodes folder
        N_EP_DIGITS = 2  # Nr of digits of each ep. folder name
        EP_DATA = "ep_data.csv"  # File name for each episode data
        EP_AGENT = "ep_agent.pth"  # File name for each agent in the episode data

    class Evals:
        FOLDER = "evaluations"  # Name of the evaluations folder
        EVAL_AGENT = "ev_agent.pth"  # Name of the evaluated agent
        EVALUATION = "eval.csv"  # Suffix of the evaluation file


@dataclass
class FileLocationManager:
    """This class uses the Naming convention to provide path handling utilities."""

    run_folder: str

    @staticmethod
    def from_nested_folders(root, project, group, run) -> FileLocationManager:
        return FileLocationManager(os.path.join(root, project, group, run))

    @property
    def episode_folder(self) -> str:
        """Root folder of the episodes."""
        return os.path.join(self.run_folder, Naming.Episodes.FOLDER)

    @property
    def evals_folder(self) -> str:
        """Root folder of the evaluations."""
        return os.path.join(self.run_folder, Naming.Evals.FOLDER)

    @property
    def metadata_file_path(self) -> str:
        return os.path.join(self.run_folder, Naming.METADATA)

    @property
    def config_file_path(self) -> str:
        return os.path.join(self.run_folder, Naming.CONFIG)

    @property
    def reward_file_path(self) -> str:
        return os.path.join(self.run_folder, Naming.REWARDS)

    def make_dirs(self) -> None:
        """Make sure that the target directories exist."""
        make_dir(self.run_folder)
        make_dir(self.episode_folder)
        make_dir(self.evals_folder)

    @staticmethod
    def episode_naming(ep_index: int) -> str:
        return f"{ep_index:0{Naming.Episodes.N_EP_DIGITS}d}"

    def make_episode_dir(self, ep_index: int) -> str:
        """Make a folder for the episode. Name is padded with zeros to n digits"""
        directory = os.path.join(
            self.episode_folder, self.episode_naming(ep_index=ep_index)
        )
        make_dir(directory)
        return directory

    def get_available_episodes(self) -> List[int]:
        """Return the integers of the episode logs available."""
        return [int(i) for i in os.listdir(self.episode_folder)]

    def get_episode_dir(self, ep_index: int) -> str:
        """Make a folder for the episode. Name is padded with zeros to n digits. Raises an error if the dir doesn't
        exist."""
        directory = os.path.join(
            self.episode_folder, self.episode_naming(ep_index=ep_index)
        )

        if not os.path.exists(directory):
            raise ValueError(f"Requested episode {ep_index} directory doesn't exist.")

        return directory

    def get_episodic_agent_filepaths(self) -> List[str]:
        file_paths = []
        for i in self.get_available_episodes():
            ep_dir = self.get_episode_dir(ep_index=i)
            file_paths.append(os.path.join(ep_dir, Naming.Episodes.EP_AGENT))
        return file_paths

    def generate_episodic_agent_filepaths(self) -> Iterator[str]:
        for i in self.get_available_episodes():
            ep_dir = self.get_episode_dir(ep_index=i)
            yield os.path.join(ep_dir, Naming.Episodes.EP_AGENT)

    def get_final_agent_file_path(self) -> str:
        return os.path.join(self.evals_folder, Naming.Evals.EVAL_AGENT)
