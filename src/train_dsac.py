import toml
import utils
from agents import DSACAgent
from data_management.log_manager import LogManager
from environments.env_phlab import PhlabEnv
from environments.episode_data import EpisodeData
from train_and_eval import train, evaluate
from typing import Dict


def main():
    """Run a Pool of processes that run the training in parallel."""

    # Load config
    config = toml.load("config.toml")

    # Fetch some settings
    n_episodes: int = config["run"]["n_episodes"]
    use_cuda: bool = config["run"]["use_cuda"]
    seed: int = config["run"]["seed"]

    # Set RNG seed
    utils.control_randomness(seed=seed)

    # Pytorch device:
    device = utils.get_device(use_cuda=use_cuda)

    # Log manager
    log_manager = LogManager(config=config)

    # Instantiate environment
    env = PhlabEnv(config=config)

    # Instantiate agent
    agent = DSACAgent(
        config=config["agent"],
        device=device,
        obs_dim=env.obs_dim,
        action_dim=env.action_dim,
    )

    # Run the training.
    train(n_episodes=n_episodes, agent=agent, env=env, log_manager=log_manager)

    # Run the evaluation
    evals: Dict[str, EpisodeData] = evaluate(env=env, agent=agent)

    # Save the evaluations
    log_manager.log_evals(agent=agent, evaluations=evals)


if __name__ == "__main__":
    main()
