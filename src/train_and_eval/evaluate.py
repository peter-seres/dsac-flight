from typing import Dict
import numpy as np
from signals import SmoothedStepSequence, Const
from environments import BaseEnv, BaseEpData, PhlabEnv, PhlabEpData
from agents import BaseAgent


def evaluate(env: PhlabEnv, agent: BaseAgent) -> Dict[str, PhlabEpData]:
    return {
        "dally": evaluate_dally(env=env, agent=agent),
        "pitch_up": evaluate_pitch_up(env=env, agent=agent),
    }


def evaluate_dally(env: PhlabEnv, agent: BaseAgent) -> PhlabEpData:
    """Evaluate the PHLAB on a number of reference signal tasks."""

    # Time of eval episode
    T = 80.0  # seconds

    # Pitch setpoints:
    p_sw = 8.0
    p_sp = [20, 10, 0, -15, 0]
    p_sp_t = [0, 16, 36, 56, 70]

    # Roll setpoints:
    r_sw = 4.0
    r_sp = [30, 0, -30, 0, 30, 0, -30]
    r_sp_t = [8, 18, 28, 38, 48, 58, 68]

    # Set env time:
    env.t_max = T

    # Signals
    theta = SmoothedStepSequence(
        smooth_width=p_sw, times=p_sp_t, amplitudes=np.deg2rad(p_sp)
    )
    phi = SmoothedStepSequence(
        smooth_width=r_sw, times=r_sp_t, amplitudes=np.deg2rad(r_sp)
    )
    beta = Const(0.0)

    # Set references:
    env.set_references(theta_ref=theta, phi_ref=phi, beta_ref=beta)

    # Make an empty episode data
    ep_data = PhlabEpData()

    # Pass in the episode data as a reference
    evaluate_single(env=env, agent=agent, ep_data=ep_data)

    return ep_data


def evaluate_pitch_up(env: PhlabEnv, agent: BaseAgent) -> PhlabEpData:
    """Evaluate the PHLAB on a number of reference signal tasks."""

    # Time of eval episode
    T = 100.0  # seconds

    # Pitch setpoints:
    p_sw = 8.0
    p_sp = [20, 0]
    p_sp_t = [0, 80]

    # Set env time:
    env.t_max = T

    # Signals
    theta = SmoothedStepSequence(
        smooth_width=p_sw, times=p_sp_t, amplitudes=np.deg2rad(p_sp)
    )
    phi = Const(0.0)
    beta = Const(0.0)

    # Set references:
    env.set_references(theta_ref=theta, phi_ref=phi, beta_ref=beta)

    # Make an empty episode data
    ep_data = PhlabEpData()

    # Pass in the episode data as a reference
    evaluate_single(env=env, agent=agent, ep_data=ep_data)

    return ep_data


def evaluate_single(env: BaseEnv, agent: BaseAgent, ep_data: BaseEpData) -> BaseEpData:
    """Run a single episode without training and save the time-series to an episode data object."""

    # Reset episode. References must be set already!
    state, info = env.reset(is_eval=True)

    episode_reward = 0
    while True:

        # Choose action
        action = agent.act_greedy(state)

        # Step the environment
        next_state, reward, done, info = env.step(action)

        # Push the data to the episodic data collector
        ep_data.push(info)

        # Store reward
        episode_reward += reward

        # Swap the states
        state = next_state

        # Break when the episode is done
        if done:
            break

    return ep_data
