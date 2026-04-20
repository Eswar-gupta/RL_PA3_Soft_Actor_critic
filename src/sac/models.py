"""Neural network modules for SAC."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import torch
from torch import nn


LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0
EPS = 1e-6


def build_mlp(input_dim: int, hidden_sizes: Iterable[int], output_dim: int, activation: type[nn.Module] = nn.ReLU) -> nn.Sequential:
    """Create a feed-forward network with ReLU hidden layers."""

    layers: list[nn.Module] = []
    last_dim = int(input_dim)
    for hidden_dim in hidden_sizes:
        layers.append(nn.Linear(last_dim, int(hidden_dim)))
        layers.append(activation())
        last_dim = int(hidden_dim)
    layers.append(nn.Linear(last_dim, int(output_dim)))
    return nn.Sequential(*layers)


class SquashedGaussianActor(nn.Module):
    """Gaussian policy with Tanh squashing and optional action scaling."""

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        hidden_sizes: Iterable[int] = (256, 256),
        action_low: np.ndarray | float | None = None,
        action_high: np.ndarray | float | None = None,
    ) -> None:
        super().__init__()
        hidden_sizes = tuple(int(size) for size in hidden_sizes)
        if not hidden_sizes:
            raise ValueError("hidden_sizes must not be empty")

        self.backbone = build_mlp(observation_dim, hidden_sizes, hidden_sizes[-1])
        self.mu_layer = nn.Linear(hidden_sizes[-1], action_dim)
        self.log_std_layer = nn.Linear(hidden_sizes[-1], action_dim)

        low, high = self._resolve_action_bounds(action_dim, action_low, action_high)
        action_scale = (high - low) / 2.0
        action_bias = (high + low) / 2.0
        self.register_buffer("action_scale", torch.as_tensor(action_scale, dtype=torch.float32))
        self.register_buffer("action_bias", torch.as_tensor(action_bias, dtype=torch.float32))

    @staticmethod
    def _resolve_action_bounds(
        action_dim: int,
        action_low: np.ndarray | float | None,
        action_high: np.ndarray | float | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        if action_low is None or action_high is None:
            low = -np.ones(action_dim, dtype=np.float32)
            high = np.ones(action_dim, dtype=np.float32)
            return low, high

        low = np.asarray(action_low, dtype=np.float32)
        high = np.asarray(action_high, dtype=np.float32)
        if low.shape == ():
            low = np.full(action_dim, float(low), dtype=np.float32)
        if high.shape == ():
            high = np.full(action_dim, float(high), dtype=np.float32)
        if low.shape != (action_dim,) or high.shape != (action_dim,):
            raise ValueError("action bounds must match the action dimension")
        return low, high

    def forward(self, observation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(observation)
        mean = self.mu_layer(features)
        log_std = self.log_std_layer(features).clamp(LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def sample(self, observation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, log_std = self.forward(observation)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        pre_tanh_action = normal.rsample()
        tanh_action = torch.tanh(pre_tanh_action)
        action = tanh_action * self.action_scale + self.action_bias

        log_prob = normal.log_prob(pre_tanh_action)
        log_prob -= torch.log(self.action_scale + EPS)
        log_prob -= torch.log(1.0 - tanh_action.pow(2) + EPS)
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        return action, log_prob, mean

    def deterministic(self, observation: torch.Tensor) -> torch.Tensor:
        mean, _ = self.forward(observation)
        return torch.tanh(mean) * self.action_scale + self.action_bias


class QNetwork(nn.Module):
    """State-action value network."""

    def __init__(self, observation_dim: int, action_dim: int, hidden_sizes: Iterable[int] = (256, 256)) -> None:
        super().__init__()
        hidden_sizes = tuple(int(size) for size in hidden_sizes)
        if not hidden_sizes:
            raise ValueError("hidden_sizes must not be empty")

        self.network = build_mlp(observation_dim + action_dim, hidden_sizes, hidden_sizes[-1])
        self.output_layer = nn.Linear(hidden_sizes[-1], 1)

    def forward(self, observation: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([observation, action], dim=-1)
        features = self.network(x)
        return self.output_layer(features)

