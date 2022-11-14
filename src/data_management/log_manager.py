import os
from typing import Dict
from agents import BaseAgent
from data_management.file_locations import FileLocationManager, Naming
from data_management.log_json import JSONLogger
from data_management.log_utils import RunMetaData, save_dict_to_toml
from environments.episode_data import EpisodeData


class LogManager(FileLocationManager):
    """A LogManager object that targets a local folder and saves data to that folder."""

    def __init__(self, config: dict):
        # Save the config.
        self.config = config

        # Fetch the save directory
        self.save_dir = config["run"]["save_dir"]
        self.run_name = config["run"]["run_name"]

        # Initialize the File Location Manager
        super().__init__(run_folder=os.path.join(self.save_dir, self.run_name))

        # Make the target folders:
        self.make_dirs()

        # Make a JSON reward logger:
        self.reward_logger = JSONLogger(file_path=self.reward_file_path)

        print(f"Log manager initialized. Saving logs to: {self.run_folder}")

    def log_config(self) -> None:
        """Save the config dictionary to TOML."""
        save_dict_to_toml(d=self.config, file_path=self.config_file_path)

    def log_metadata(self, agent: str) -> None:
        """Save the metadata to TOML."""

        use_cuda = self.config["run"]["use_cuda"]
        seed = self.config["run"]["seed"]

        metadata = RunMetaData(agent=agent, use_cuda=use_cuda, seed=seed)
        metadata.save_to_toml(file_path=self.metadata_file_path)

    def log_reward(self, reward: float) -> None:
        """ " Save end-of-episode reward to JSON"""
        self.reward_logger.save(reward=reward)

    def log_episode(
        self, ep_index: int, reward: float, ep_data: EpisodeData, ep_agent: BaseAgent
    ) -> None:

        self.log_reward(reward=reward)

        # Make a folder for the episode
        ep_folder = self.make_episode_dir(ep_index=ep_index)

        # Save the episode to .CSV:
        file_path = os.path.join(ep_folder, Naming.Episodes.EP_DATA)
        ep_data.to_pandas(convert_to_deg=True).to_csv(file_path)

        # Save the agent to .PTH:
        file_path = os.path.join(ep_folder, Naming.Episodes.EP_AGENT)
        ep_agent.save_to_file(file_path=file_path)

    def log_evals(self, agent: BaseAgent, evaluations: Dict[str, EpisodeData]):

        # Save the evaluations in the dictionary
        for key, evaluation_ep_data in evaluations.items():
            file_path = os.path.join(
                self.evals_folder, f"{key}_{Naming.Evals.EVALUATION}"
            )
            evaluation_ep_data.to_pandas(convert_to_deg=True).to_csv(file_path)

        # Save the agent:
        file_path_eval_agent = os.path.join(self.evals_folder, Naming.Evals.EVAL_AGENT)
        agent.save_to_file(file_path=file_path_eval_agent)

        print(f"\n Saved evaluations to: {self.evals_folder}")
