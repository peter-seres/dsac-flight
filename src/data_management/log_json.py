import json
import os.path
import numpy as np


class JSONLogger:
    """This class makes a JSON file and allows the user to periodically save data to the file."""

    def __init__(self, file_path: str):
        self.handle_non_json(file_path)

        # If the file already exists just issue a warning that we're going to overwrite it:
        if os.path.isfile(file_path):
            print(f"Target file '{file_path}' already exists and will be overwritten.")

        # Store the file path:
        self.file_path = file_path

        with open(self.file_path, "w") as file_target:
            json.dump(
                {"reward": []},
                file_target,
                indent=4,
                sort_keys=True,
            )

    def save(self, reward: float) -> None:
        reward = float(reward)

        # Read context
        with open(self.file_path, "r") as file_target:

            # Load the json
            d = json.load(file_target)

            # Append the data
            d["reward"].append(reward)

        # Write context
        with open(self.file_path, "w") as file_target:

            # Save the json
            json.dump(d, file_target)

    @staticmethod
    def handle_non_json(file_path: str) -> None:
        """Handle non-JSON file target"""
        if not file_path.endswith(".json"):
            raise ValueError(f"Expected .json file extension. Got: {file_path}.")

    @classmethod
    def load_from_json(cls, file_path: str) -> dict:
        """Load a target JSON file to a dictionary"""
        cls.handle_non_json(file_path=file_path)
        with open(file_path, "r") as file_target:
            return json.load(file_target)

    @classmethod
    def load_to_numpy(cls, file_path: str) -> (np.ndarray, np.ndarray):
        """Load a target JSON file to two numpy arrays"""
        d = cls.load_from_json(file_path=file_path)
        return np.array(d["reward"])
