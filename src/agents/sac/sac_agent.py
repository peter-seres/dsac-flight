from __future__ import annotations
import os
from typing import Tuple
import numpy as np
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from agents.base_agent import BaseAgent
from agents.sac.critic import QNet
from agents.sac.actor import NormalPolicyNet
from agents.experience_replay import Batch, ReplayBuffer, Transition

CLIP_GRAD = 1.0
CAPS_STD = 0.05


class SACAgent(BaseAgent):
    def __init__(
        self, device: torch.device, config: dict, obs_dim: int, action_dim: int
    ):
        super().__init__(
            device=device, config=config, obs_dim=obs_dim, action_dim=action_dim
        )

        # Hyperparameters
        self.batch_size = config["batch_size"]
        self.gamma = config["gamma"]
        self.lr0 = config["lr"]
        self.lr = config["lr"]
        self.polyak_step_size = config["polyak_step_size"]
        self.n_hidden_layers = config["n_hidden_layers"]
        self.n_hidden_units = config["n_hidden_units"]
        self.update_every = config.get("update_every") or 1
        self.LR_ZERO_AT = config.get("lr_zero_at") or 1e6

        # New CAPS params:
        self.lambda_t = config.get("lambda_t") or 0.0
        self.lambda_s = config.get("lambda_s") or 0.0

        # Replay buffer
        self.memory = ReplayBuffer(
            buffer_size=config["buffer_size"],
            device=self.device,
        )

        # Actor
        self.policy = NormalPolicyNet(
            obs_dim=obs_dim,
            action_dim=action_dim,
            n_hidden_layers=self.n_hidden_layers,
            n_hidden_units=self.n_hidden_units,
        ).to(device=self.device)

        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=self.lr)

        self.Q1 = QNet(
            obs_dim=obs_dim,
            action_dim=action_dim,
            n_hidden_layers=self.n_hidden_layers,
            n_hidden_units=self.n_hidden_units,
        ).to(device=self.device)

        self.Q1_target = QNet(
            obs_dim=obs_dim,
            action_dim=action_dim,
            n_hidden_layers=self.n_hidden_layers,
            n_hidden_units=self.n_hidden_units,
        ).to(device=self.device)

        self.Q1_target.load_state_dict(self.Q1.state_dict())
        self.Q1_optimizer = optim.Adam(self.Q1.parameters(), lr=self.lr)

        self.Q2 = QNet(
            obs_dim=obs_dim,
            action_dim=action_dim,
            n_hidden_layers=self.n_hidden_layers,
            n_hidden_units=self.n_hidden_units,
        ).to(device=self.device)

        self.Q2_target = QNet(
            obs_dim=obs_dim,
            action_dim=action_dim,
            n_hidden_layers=self.n_hidden_layers,
            n_hidden_units=self.n_hidden_units,
        ).to(device=self.device)

        self.Q2_target.load_state_dict(self.Q2.state_dict())
        self.Q2_optimizer = optim.Adam(self.Q2.parameters(), lr=self.lr)

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

            if self.update_counter % self.update_every == 0:
                self.update_networks(self.memory.sample(self.batch_size))

    def get_eta(self):
        return self.log_eta.exp()

    def sample(
        self, state: torch.tensor, reparameterize: bool
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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

    def update_networks(self, b: Batch) -> None:
        s, a, r, next_state, d = b
        eta = self.get_eta()

        with torch.no_grad():
            next_action, next_log_pi = self.sample(next_state, reparameterize=False)
            targets = r + self.gamma * (1 - d) * (
                torch.min(
                    self.Q1_target(next_state, next_action),
                    self.Q2_target(next_state, next_action),
                )
                - eta * next_log_pi
            )

        # Critic Loss 1
        Q1_predictions = self.Q1(s, a)
        Q1_loss = torch.mean((Q1_predictions - targets) ** 2)

        # Optimize Q1
        self.Q1_optimizer.zero_grad()
        Q1_loss.backward()
        self.clip_gradient(net=self.Q1)
        self.Q1_optimizer.step()

        # Critic loss 2
        Q2_predictions = self.Q2(s, a)
        Q2_loss = torch.mean((Q2_predictions - targets) ** 2)

        # Optimize Q2
        self.Q2_optimizer.zero_grad()
        Q2_loss.backward()
        self.clip_gradient(net=self.Q2)
        self.Q2_optimizer.step()

        # Disable the gradients of the critic networks:
        for param in self.Q1.parameters():
            param.requires_grad = False
        for param in self.Q2.parameters():
            param.requires_grad = False

        # ____ Calculate policy loss____

        # Action at current state: a = pi(s)
        new_action, log_pi = self.sample(s, reparameterize=True)

        # CAPS spatial smoothness: distance between a = pi(s) and a* = pi(s*) where s* ~ normal(s, std_dev)
        new_action_deterministic = self.policy.get_mean(states=s)
        action_nearby = self.policy.get_mean(states=torch.normal(mean=s, std=CAPS_STD))
        space_smoothness_loss = torch.mean(
            (new_action_deterministic - action_nearby) ** 2
        )
        space_smoothness_loss *= self.lambda_s / new_action.shape[0]

        # CAPS temporal smoothness: distance between a = pi(s) and a' = pi(s')
        time_smoothness_loss = torch.mean(
            (new_action - next_action) ** 2
        )  # next_action is a_next = pi(s_next)
        time_smoothness_loss *= self.lambda_t / new_action.shape[0]

        # Critic value:
        Q1 = self.Q1(s, new_action)
        Q2 = self.Q2(s, new_action)
        Q = torch.min(Q1, Q2)

        # Final policy loss
        policy_loss = -torch.mean(
            Q - eta * log_pi - space_smoothness_loss - time_smoothness_loss
        )

        # Optimize the actor
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.clip_gradient(net=self.policy)
        self.policy_optimizer.step()

        # Enable the gradients of the critic networks:
        for param in self.Q1.parameters():
            param.requires_grad = True
        for param in self.Q2.parameters():
            param.requires_grad = True

        # Temperature loss
        eta_loss = -(self.log_eta * (log_pi + self.target_entropy).detach()).mean()

        # Optimize temperature
        self.eta_optimizer.zero_grad()
        eta_loss.backward()
        self.eta_optimizer.step()

        # Soft update the target networks:
        with torch.no_grad():
            self.polyak_update(old_net=self.Q1_target, new_net=self.Q1)
            self.polyak_update(old_net=self.Q2_target, new_net=self.Q2)

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

    def get_episode_log(self) -> dict:
        return {"eta:": self.get_eta()}

    def set_eval(self):
        """Set the policy the eval mode."""
        self.policy.eval()

    def save_to_file(self, file_path: str) -> None:
        torch.save(
            {
                "Q1": self.Q1.state_dict(),
                "Q2": self.Q2.state_dict(),
                "Q1_target": self.Q1_target.state_dict(),
                "Q2_target": self.Q2_target.state_dict(),
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
    ) -> SACAgent:

        # Load the saved network parameters
        params = torch.load(file_path, map_location=device)

        # Instantiate a new agent
        agent = SACAgent(
            device=device, config=config, obs_dim=obs_dim, action_dim=action_dim
        )

        # Override params
        agent.Q1.load_state_dict(params["Q1"])
        agent.Q2.load_state_dict(params["Q2"])
        agent.Q1_target.load_state_dict(params["Q1_target"])
        agent.Q2_target.load_state_dict(params["Q2_target"])
        agent.policy.load_state_dict(params["policy"])
        agent.log_eta = params["log_eta"]

        return agent
