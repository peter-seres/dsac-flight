"""
Make an empty EpisodeData object and push random data to it. Save to a .csv file
"""

import numpy as np
from src.environments.episode_data import EpisodeData


def test_episode_data():

    ep_data = EpisodeData()

    for t in np.arange(0.0, 15.0, 0.01):
        info = {
            "t": float(t),
            "x": np.random.rand(12),
            "u": np.random.rand(3),
            "ref": np.zeros(3),
            "var": np.array([0.0, 0.0]),
            "reward": float(10.0),
        }

        ep_data.push(info=info)

    ep_data.to_pandas(convert_to_deg=True).to_csv("test.csv")


if __name__ == "__main__":
    test_episode_data()
