from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Optional
import toml
import datetime
import pathlib
import git


def get_git_commit(shorten: bool = True) -> str:
    """Gets the current version control commit hash."""
    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha
    return sha[:7] if shorten else sha


def make_dir(path: str) -> None:
    """Make a directory at the target path."""
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)


def get_current_time():
    """Fetch current time."""
    return datetime.datetime.now().strftime("%y-%m-%d %H:%M:%S")


def save_dict_to_toml(d: dict, file_path: str) -> None:
    """Save a dictionary to a .toml file"""
    with open(file_path, "w") as f:
        toml.dump(d, f)


@dataclass
class Saveable:
    """A dataclass that can be saved to a toml file."""

    def to_dict(self) -> dict:
        return {
            k: (v.to_dict() if hasattr(v, "__dict__") else v)
            for k, v in self.__dict__.items()
        }

    def save_to_toml(self, file_path: str) -> None:
        save_dict_to_toml(self.to_dict(), file_path=file_path)


@dataclass
class RunMetaData(Saveable):
    agent: str
    use_cuda: bool
    seed: Optional[int]
    computer: str = os.environ["COMPUTERNAME"]
    commit: str = get_git_commit()
    time: str = get_current_time()
