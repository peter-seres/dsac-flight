"""
This module implements the distributional critic based on the Implicit Quantile Network approach.
The DNN approximates the return distribution Z(s, a; tau) implicity using tau quantiles.
"""

import torch
import torch.nn as nn
from agents.mlp import make_mlp


class ZNet(nn.Module):
    """Wrapper class around IQN, that has only 1 output tensor for Z(s, a)"""

    def __init__(
        self,
        n_states: int,
        n_actions: int,
        n_hidden_layers: int,
        n_hidden_units: int,
        n_cos: int,
        device: torch.device,
    ):
        super().__init__()
        self.device = device

        self.iqn = IQN(
            n_inputs=n_states + n_actions,
            n_outputs=1,
            n_hidden_layers=n_hidden_layers,
            n_hidden_units=n_hidden_units,
            embedding_size=n_cos,
            device=device,
        ).to(self.device)

    @staticmethod
    def generate_taus(
        batch_size: int, n_taus: int, device: torch.device
    ) -> torch.Tensor:
        with torch.no_grad():
            taus = torch.rand(batch_size, n_taus, device=device)  # Size(B, N)
            return taus  # Size (B, N)

    def get_expectation(
        self, s: torch.Tensor, a: torch.Tensor, n_taus: int = 32
    ) -> torch.Tensor:
        # Batch size:
        B = s.shape[0]

        # Generate taus to evaluate expectation:
        taus = self.generate_taus(
            batch_size=B, n_taus=n_taus, device=self.device
        )  # Shape (B, N)

        # Pass through the network:
        z = self(s, a, taus=taus)  # Shape (B, N)

        # Calculate the expectation:
        q = z.mean(dim=1)  # Shape (B, 1)

        return q

    def get_variance(
        self, s: torch.Tensor, a: torch.Tensor, n_taus: int = 32
    ) -> torch.Tensor:
        # Batch size:
        B = s.shape[0]

        # Generate taus to evaluate expectation:
        taus = self.generate_taus(
            batch_size=B, n_taus=n_taus, device=self.device
        )  # Shape (B, N)

        # Pass through the network:
        z: torch.Tensor = self(s, a, taus=taus)  # Shape (B, N)

        var = torch.var(
            input=z,
            dim=1,
        )

        return var

    def forward(self, s: torch.Tensor, a: torch.Tensor, taus: torch.Tensor):
        # Merge states and actions
        x = torch.cat([s, a], dim=1)

        # Pass through the network
        taus = taus.unsqueeze(-1)  # Taus must be of shape (B, N, 1)
        z = self.iqn(x, taus)  # Shape (B, N, 1)

        # Remove the action dimensions
        z = z.squeeze(2)  # Shape (B, N)

        return z


class IQN(nn.Module):
    """Implicit quantile netural network for discrete action spaces."""

    def __init__(
        self,
        n_inputs: int,
        n_outputs: int,
        embedding_size: int,
        n_hidden_layers: int,
        n_hidden_units: int,
        device: torch.device,
    ):
        self.device = device

        # State input layer
        super().__init__()

        # Save sizes for Tensor reshaping
        self.S = n_inputs
        self.A = n_outputs
        self.C = self.embedding_size = embedding_size
        self.H = n_hidden_units

        # The states are passed through the input layer
        self.input_layer = nn.Sequential(
            nn.Linear(n_inputs, n_hidden_units),
            nn.LayerNorm(n_hidden_units),
            nn.ReLU(),
        )

        # Quantile embedding layer
        self.const_pi_vec = (
            torch.arange(start=0, end=self.embedding_size, device=self.device)
            * torch.pi
        )
        self.embedding_layer = nn.Sequential(
            nn.Linear(embedding_size, n_hidden_units),
            nn.LayerNorm(n_hidden_units),
            nn.Sigmoid(),
        )

        # Hidden Layers
        self.hidden_layers = make_mlp(
            num_in=n_hidden_units,
            num_out=n_outputs,
            final_activation=None,
            n_hidden_layers=n_hidden_layers,
            n_hidden_units=n_hidden_units,
        )

    def forward(self, x: torch.Tensor, taus: torch.Tensor) -> torch.Tensor:
        """
        Quantile calculation depending on the number of tau
        input x must be of shape (batch_size (B), state_size (S))
        input taus must be of shape (batch_size (B), nr of taus (N), 1)
        :returns:
            quantiles [batch_size (B), num_tau (N), action_size (A)]
        """

        A = self.A  # Action size
        B = x.shape[0]  # Batch size
        N = taus.shape[1]  # Number of taus
        C = self.C  # Embedding layer size
        H = self.H  # Hidden layer size

        # Pass the states through input layers
        x = self.input_layer(x)  # Size (B, H)

        # Pass through the cosine embedding layer
        cos = torch.cos(taus * self.const_pi_vec)  # Size (B, N, C)
        cos = cos.view(B * N, C)  # Size (B * N, C)
        cos_out = self.embedding_layer(cos)  # Size (B * N, H)

        # Do some reshaping to match the two input tensors
        cos_out_reshaped = cos_out.view(B, N, H)  # Size (B, N, H)

        # Insert a dimension to match the cosine output
        x_reshaped = x.unsqueeze(1)  # Size (B, 1, H)

        # Merge
        h = torch.mul(x_reshaped, cos_out_reshaped)  # Size (B, N, H)

        # Reshape to pass through the remaining layers:
        h = h.view(B * N, H)  # Size (B * N, H)
        output = self.hidden_layers(h)  # Size (B * N, A)

        # Reshape to separate batch size and taus
        output = output.view(B, N, A)  # Size (B, N, A)

        return output

    def get_q_values(self, x: torch.Tensor, n_taus: int):
        """
        Calculate expected return using n_taus (N) quantiles.
        :return: Q values of shape (batch_size (B), n_actions (A))
        """
        taus = self.generate_taus(
            batch_size=x.shape[0], n_taus=n_taus
        )  # Size (B, N, 1)
        quantiles = self.forward(x, taus=taus)  # Size (B, N, A)
        expectation = quantiles.mean(dim=1)  # Size (B, A)
        return expectation
