from __future__ import annotations
import os
import numpy as np
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from agents.dsac.critic import ZNet
from agents.dsac.risk_distortions import DistortionFn, distortion_functions
from agents.base_agent import BaseAgent
from agents.sac.actor import NormalPolicyNet
from agents.experience_replay import Batch, ReplayBuffer, Transition

CLIP_GRAD = 1.0
CAPS_STD = 0.05


def calculate_huber_loss(td_error: torch.Tensor, k: float = 1.0) -> torch.Tensor:
    """Calculate huber loss element-wisely depending on kappa k."""
    loss = torch.where(
        td_error.abs() <= k,
        0.5 * td_error.pow(2),
        k * (td_error.abs() - 0.5 * k),
    )

    return loss


def quantile_huber_loss(
    target: torch.Tensor,
    prediction: torch.Tensor,
    taus: torch.Tensor,
    kappa: float = 1.0,
):

    # TD-error
    td_error = target - prediction  # Shape (B, N)

    # Calculate huber loss for each sample
    huber_l = calculate_huber_loss(td_error=td_error, k=kappa)  # Shape (B, N)

    # Tau-quantile huber loss
    rho = abs(taus - (td_error.detach() < 0).float()) * huber_l / kappa  # Shape (B, N)
    loss = rho.sum(dim=1).mean()  # Shape 1

    return loss


class DSACAgent(BaseAgent):
    def __init__(
        self, device: torch.device, config: dict, obs_dim: int, action_dim: int
    ):
        super().__init__(
            device=device, config=config, obs_dim=obs_dim, action_dim=action_dim
        )

        # Hyperparameters
        self.batch_size = config["batch_size"]
        self.gamma = config["gamma"]
        self.lr = config["lr"]
        self.lr0 = config["lr"]
        self.polyak_step_size = config["polyak_step_size"]
        self.n_hidden_layers = config["n_hidden_layers"]
        self.n_hidden_units = config["n_hidden_units"]
        self.update_every = config.get("update_every") or 1
        self.LR_ZERO_AT = config.get("lr_zero_at") or 1e6

        # IQN params:
        self.n_taus = config["n_taus"]
        self.n_taus_exp = config["n_taus_exp"]

        # New CAPS params:
        self.lambda_t = config.get("lambda_t") or 0.0
        self.lambda_s = config.get("lambda_s") or 0.0

        # Risk distortion parameter.
        self.risk_measure = config.get("risk_measure") or 1.0
        risk_type = config.get("risk_distortion") or "neutral"
        self.risk_distortion_fn: DistortionFn = distortion_functions.get(risk_type)

        # Replay buffer
        self.memory = ReplayBuffer(
            buffer_size=config["buffer_size"], device=self.device
        )

        # Actor
        self.policy = NormalPolicyNet(
            obs_dim=obs_dim,
            action_dim=action_dim,
            n_hidden_layers=self.n_hidden_layers,
            n_hidden_units=self.n_hidden_units,
        ).to(device=self.device)
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=self.lr)

        self.Z1 = ZNet(
            n_states=obs_dim,
            n_actions=action_dim,
            n_hidden_layers=self.n_hidden_layers,
            n_hidden_units=self.n_hidden_units,
            n_cos=config["n_cos"],
            device=self.device,
        ).to(device=self.device)
        self.Z1_target = ZNet(
            n_states=obs_dim,
            n_actions=action_dim,
            n_hidden_layers=self.n_hidden_layers,
            n_hidden_units=self.n_hidden_units,
            n_cos=config["n_cos"],
            device=self.device,
        ).to(device=self.device)
        self.Z1_target.load_state_dict(self.Z1.state_dict())
        self.Z1_optimizer = optim.Adam(self.Z1.parameters(), lr=self.lr)

        self.Z2 = ZNet(
            n_states=obs_dim,
            n_actions=action_dim,
            n_hidden_layers=self.n_hidden_layers,
            n_hidden_units=self.n_hidden_units,
            n_cos=config["n_cos"],
            device=self.device,
        ).to(device=self.device)
        self.Z2_target = ZNet(
            n_states=obs_dim,
            n_actions=action_dim,
            n_hidden_layers=self.n_hidden_layers,
            n_hidden_units=self.n_hidden_units,
            n_cos=config["n_cos"],
            device=self.device,
        ).to(device=self.device)
        self.Z2_target.load_state_dict(self.Z2.state_dict())
        self.Z2_optimizer = optim.Adam(self.Z2.parameters(), lr=self.lr)

        # Adaptive temperature:
        self.target_entropy = -np.prod(action_dim).item()
        self.log_eta = torch.zeros(1, requires_grad=True, device=self.device)
        self.eta_optimizer = optim.Adam([self.log_eta], lr=self.lr)

        # Count updates:
        self.update_counter = 0

        # Episode counter
        self.episode_counter = 0

    def every_sample_update_callback(self) -> None:
        """Linearly decay learning rate from self.lr0 to 0 at N_SAMPLES."""
        self.episode_counter += 1
        self.lr = self.lr0 - (self.lr0 / self.LR_ZERO_AT) * self.episode_counter
        self.lr = max(0.0, self.lr)

    def update(self, state, action, reward, next_state, done):
        # Store transition
        self.memory.push(Transition(state, action, reward, next_state, done))
        if self.memory.ready_for(self.batch_size):
            self.update_counter += 1
            losses = self.update_networks(self.memory.sample(self.batch_size))
            return losses
        return None

    def get_eta(self):
        return self.log_eta.exp()

    def sample(self, state: torch.tensor, reparameterize: bool) -> tuple:
        """Stochastic action: sample from the gaussian and calculate log probs."""

        # Normal distribution:
        normal = self.policy(state)

        # Sample actions:
        u = normal.rsample() if reparameterize else normal.sample()

        # Tanh squash actions
        a = torch.tanh(u)

        # Calculate log probabilities:
        log_pi = normal.log_prob(u) - (2 * (np.log(2) - u - F.softplus(-2 * u))).sum(
            dim=1
        )
        return a, log_pi

    def polyak_update(self, old_net: nn.Module, new_net: nn.Module) -> None:
        for old_param, new_param in zip(old_net.parameters(), new_net.parameters()):
            old_param.data.copy_(
                old_param.data * self.polyak_step_size
                + new_param.data * (1 - self.polyak_step_size)
            )

    def update_networks(self, b: Batch) -> np.ndarray:
        # Unpack the mini-batch
        s, a, r, next_state, d = b

        # Tensor sizes:
        B = self.batch_size
        N = self.n_taus

        # Fetch the adaptive temperature
        eta = self.get_eta()

        # Calculate TD-targets:
        with torch.no_grad():
            next_action, next_log_pi = self.sample(next_state, reparameterize=False)

            # Generate the taus separately due to min operator
            taus_i = ZNet.generate_taus(batch_size=B, n_taus=N, device=self.device)
            taus_j = ZNet.generate_taus(batch_size=B, n_taus=N, device=self.device)

            # Fetch the target for both target networks and take the minimum
            z1_target = self.Z1_target(next_state, next_action, taus_i)
            z2_target = self.Z2_target(next_state, next_action, taus_i)
            min_target = torch.min(z1_target, z2_target)

            # Soft TD-target
            entropy_term = (eta * next_log_pi).unsqueeze(-1)
            target = r + self.gamma * (1 - d) * (min_target - entropy_term)

        # Critic predictions:
        Z1_pred = self.Z1(s, a, taus_j)
        Z2_pred = self.Z2(s, a, taus_j)
        Z1_loss = quantile_huber_loss(target=target, prediction=Z1_pred, taus=taus_j)
        Z2_loss = quantile_huber_loss(target=target, prediction=Z2_pred, taus=taus_j)

        # Optimize Z1
        self.Z1_optimizer.zero_grad()
        Z1_loss.backward()
        self.clip_gradient(net=self.Z1)
        self.Z1_optimizer.step()

        # Optimize Q2
        self.Z2_optimizer.zero_grad()
        Z2_loss.backward()
        self.clip_gradient(net=self.Z2)
        self.Z2_optimizer.step()

        # Disable the gradients of the critic networks:
        for param in self.Z1.parameters():
            param.requires_grad = False
        for param in self.Z2.parameters():
            param.requires_grad = False

        # Calculate policy loss
        new_action, log_pi = self.sample(s, reparameterize=True)

        # CAPS spatial smoothness: distance between a = pi(s) and a* = pi(s*) where s* ~ normal(s, std_dev)
        new_acion_deterministic = self.policy.get_mean(states=s)
        action_nearby = self.policy.get_mean(states=torch.normal(mean=s, std=CAPS_STD))
        space_smoothness_loss = torch.mean(
            (new_acion_deterministic - action_nearby) ** 2
        )
        space_smoothness_loss *= self.lambda_s / new_action.shape[0]

        # CAPS temporal smoothness: distance between a = pi(s) and a' = pi(s')
        time_smoothness_loss = torch.mean(
            (new_action - next_action) ** 2
        )  # next_action is a_next = pi(s_next)
        time_smoothness_loss *= self.lambda_t / new_action.shape[0]

        # Critic values:

        # Generate random taus
        taus_exp = ZNet.generate_taus(
            batch_size=B, n_taus=self.n_taus_exp, device=self.device
        )

        # Distort the taus
        taus_dist = self.risk_distortion_fn(taus_exp, self.risk_measure)

        # Forward pass through the critic networks:
        Z1_r = self.Z1(s=s, a=new_action, taus=taus_dist)  # risk-distorted Z
        Z2_r = self.Z2(s=s, a=new_action, taus=taus_dist)  # risk-distorted Z

        Q1_r = Z1_r.mean(dim=1)  # Risk-distorted expectation
        Q2_r = Z2_r.mean(dim=1)  # Risk-distorted expectation

        # Take the lower value:
        Q = torch.min(Q1_r, Q2_r)  # Shape (B)

        # Loss function
        policy_loss = -torch.mean(
            Q - eta * log_pi - space_smoothness_loss - time_smoothness_loss
        )

        # Optimize the actor
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.clip_gradient(net=self.policy)
        self.policy_optimizer.step()

        # Enable the gradients of the critic networks:
        for param in self.Z1.parameters():
            param.requires_grad = True
        for param in self.Z2.parameters():
            param.requires_grad = True

        # Temperature loss
        eta_loss = -(self.log_eta * (log_pi + self.target_entropy).detach()).mean()

        # Optimize temperature
        self.eta_optimizer.zero_grad()
        eta_loss.backward()
        self.eta_optimizer.step()

        # Soft update the target networks:
        with torch.no_grad():
            self.polyak_update(old_net=self.Z1_target, new_net=self.Z1)
            self.polyak_update(old_net=self.Z2_target, new_net=self.Z2)

        losses = [float(Z1_loss), float(Z2_loss), float(policy_loss), float(eta_loss)]

        return np.array(losses)

    def act(self, state: np.array) -> np.array:
        state = torch.tensor(state, device=self.device).unsqueeze(0).float()
        action, _ = self.sample(state, reparameterize=False)
        return action.cpu().numpy()[0]

    def act_greedy(self, state: np.ndarray):
        """Deterministic action takes the mean of the gaussian policy."""
        with torch.no_grad():
            state = torch.tensor(state, device=self.device).unsqueeze(0).float()
            mean = self.policy.get_mean(state)
            return mean.detach().cpu().numpy()[0]

    def save_actor(self, save_dir: str, filename: str) -> None:
        os.makedirs(save_dir, exist_ok=True)
        torch.save(self.policy.state_dict(), os.path.join(save_dir, filename))

    def load_actor(self, save_dir: str, filename: str) -> None:
        self.policy.load_state_dict(torch.load(os.path.join(save_dir, filename)))

    @staticmethod
    def clip_gradient(net: nn.Module) -> None:
        for param in net.parameters():
            param.grad.data.clamp_(-CLIP_GRAD, CLIP_GRAD)

    def set_eval(self):
        """Set the policy the eval mode."""
        self.policy.eval()

    def save_to_file(self, file_path: str) -> None:
        torch.save(
            {
                "Z1": self.Z1.state_dict(),
                "Z2": self.Z2.state_dict(),
                "Z1_target": self.Z1_target.state_dict(),
                "Z2_target": self.Z2_target.state_dict(),
                "policy": self.policy.state_dict(),
                "log_eta": self.log_eta,
            },
            f=file_path,
        )

    @staticmethod
    def load_from_file(
        file_path: str,
        config: dict,
        device: torch.device,
        obs_dim: int,
        action_dim: int,
    ) -> DSACAgent:

        # Load the saved network parameters
        params = torch.load(file_path, map_location=device)

        # Instantiate a new agent
        agent = DSACAgent(
            device=device,
            config=config["agent"],
            obs_dim=obs_dim,
            action_dim=action_dim,
        )

        # Override params
        agent.Z1.load_state_dict(params["Z1"])
        agent.Z2.load_state_dict(params["Z2"])
        agent.Z1_target.load_state_dict(params["Z1_target"])
        agent.Z2_target.load_state_dict(params["Z2_target"])
        agent.policy.load_state_dict(params["policy"])
        agent.log_eta = params["log_eta"]

        return agent
