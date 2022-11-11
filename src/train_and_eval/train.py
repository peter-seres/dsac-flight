import numpy as np
import tqdm
from agents import BaseAgent
from environments.env_phlab import PhlabEnv
from environments.episode_data import EpisodeData
from data_management.log_manager import LogManager


def train(
    n_episodes: int,
    env: PhlabEnv,
    agent: BaseAgent,
    log_manager: LogManager,
):
    """Train the agent through n_episodes."""

    # Reward stats:
    rewards = np.zeros(n_episodes, dtype=np.float64)

    # Progress bar
    ep_iter = tqdm.tqdm(range(n_episodes), colour="cyan", unit="episode", position=0)

    # Iterate through episodes:
    for i in ep_iter:

        # Update progress bar:
        ep_iter.set_description(desc=f"Prev. return: {rewards[i - 1]:,.1f}")

        # Run the training for a single episode:
        ep_data, total_reward = train_single_episode(env=env, agent=agent)

        # Save rewards:
        rewards[i] = total_reward

        # Log the episode data:
        log_manager.log_episode(
            ep_index=i,
            reward=total_reward,
            ep_data=ep_data,
            ep_agent=agent,
        )


def train_single_episode(
    env: PhlabEnv, agent: BaseAgent
) -> (EpisodeData, float, float):

    """Returns the (episode data object, end-of-episode return, wall-clock training time)"""

    # Reset environment
    state, info = env.reset()
    total_reward = 0

    # Initialize an episode data container
    ep_data = EpisodeData()
    ep_data.push(info)

    # Make an episode progress bar
    ep_bar = tqdm.tqdm(
        desc="Current Episode",
        total=env.max_samples,
        mininterval=1.0,
        unit="sample",
        position=2,
        colour="magenta",
        leave=False,
    )

    while True:
        # callback:
        agent.every_sample_update_callback()

        # Choose action
        action = agent.act(state)

        # Step the environment
        next_state, reward, done, info = env.step(action)

        if info.get("nancrash"):
            print("NAN crash. Breaking episode without agent update")
            break

        # Accumulate reward
        total_reward += reward

        # Push the data to the episodic data collector
        ep_data.push(info=info)

        # Update the agent
        agent.update(state, action, reward, next_state, done)

        # Swap the states
        state = next_state

        # Update episode bar
        ep_bar.update(1)

        # Break when the episode is done
        if done:
            break

    return ep_data, total_reward
