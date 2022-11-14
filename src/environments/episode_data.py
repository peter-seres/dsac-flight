"""
This module implements a class that stores episode data effiently.
Implements saving to .csv and using pd.Dataframes.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from collections import deque
from typing import Deque


@dataclass(slots=True)
class EpisodeData:
    """Uses deque and slots for efficiency. to_pandas converts it into a pd.DataFrame."""

    # Time axis
    t: Deque[float] = field(default_factory=deque, repr=False)

    # Dynamic states
    x: Deque[np.ndarray] = field(default_factory=deque, repr=False)

    # Control inputs
    u: Deque[np.ndarray] = field(default_factory=deque, repr=False)

    # References
    refs: Deque[np.ndarray] = field(default_factory=deque, repr=False)

    # Variance of the critic (if applicable)
    var: Deque[np.ndarray] = field(default_factory=deque, repr=False)

    # Immediate rewards
    rewards: Deque[float] = field(default_factory=deque, repr=False)

    def push(self, info: dict):
        self.t.append(info.get("t"))
        self.x.append(info.get("x"))
        self.u.append(info.get("u"))
        self.refs.append(info.get("ref"))
        self.var.append(info.get("var"))
        self.rewards.append(info.get("reward"))

    def to_pandas(self, convert_to_deg: bool = True) -> pd.DataFrame:
        # Conver them into numpy matrices
        t = np.array([self.t]).T
        r = np.array([self.rewards]).T
        x = np.vstack(self.x)
        u = np.vstack(self.u)
        refs = np.vstack(self.refs)
        var = np.vstack(self.var)

        # Concat into a big matrix
        data = np.concatenate((t, r, x, u, refs, var), axis=1)

        # Make a dataframe
        df = pd.DataFrame(
            data,
            columns=[
                "t",
                "reward",
                "p",
                "q",
                "r",
                "V",
                "a",
                "b",
                "phi",
                "theta",
                "psi",
                "h",
                "xe",
                "ye",
                "de",
                "da",
                "dr",
                "beta_ref",
                "theta_ref",
                "phi_ref",
                "var_1",
                "var_2",
            ],
        )

        # Convert from radians to degrees:
        if convert_to_deg:
            df = df.apply(
                lambda col: np.rad2deg(col)
                if col.name
                not in ["t", "reward", "V", "h", "xe", "ye", "var_1", "var_2"]
                else col
            )

        return df
