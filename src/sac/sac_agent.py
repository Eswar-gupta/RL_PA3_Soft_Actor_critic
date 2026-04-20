"""Soft Actor-Critic agent with automatic entropy tuning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from src.replay_buffer import ReplayBatch
from src.sac.models import QNetwork, SquashedGaussianActor
from src.utils import hard_update, soft_update, to_tensor


@dataclass(slots=True)
class SACConfig:
    observation_dim: int
    action_dim: int
    action_low: np.ndarray | float | None = None
    action_high: np.ndarray | float | None = None
    hidden_sizes: tuple[int, ...] = (256, 256)
    gamma: float = 0.99
    tau: float = 0.005
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4
    target_entropy: float | None = None
    device: str | torch.device = "cpu"


class SACAgent:
    """Continuous-control SAC agent."""

    def __init__(self, config: SACConfig) -> None:
        self.config = config
        self.device = torch.device(config.device)
        self.gamma = float(config.gamma)
        self.tau = float(config.tau)
        self.target_entropy = float(config.target_entropy if config.target_entropy is not None else -config.action_dim)

        self.actor = SquashedGaussianActor(
            observation_dim=config.observation_dim,
            action_dim=config.action_dim,
            hidden_sizes=config.hidden_sizes,
            action_low=config.action_low,
            action_high=config.action_high,
        ).to(self.device)
        self.q1 = QNetwork(config.observation_dim, config.action_dim, hidden_sizes=config.hidden_sizes).to(self.device)
        self.q2 = QNetwork(config.observation_dim, config.action_dim, hidden_sizes=config.hidden_sizes).to(self.device)
        self.target_q1 = QNetwork(config.observation_dim, config.action_dim, hidden_sizes=config.hidden_sizes).to(self.device)
        self.target_q2 = QNetwork(config.observation_dim, config.action_dim, hidden_sizes=config.hidden_sizes).to(self.device)

        hard_update(self.target_q1, self.q1)
        hard_update(self.target_q2, self.q2)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config.actor_lr)
        self.critic_optimizer = torch.optim.Adam(list(self.q1.parameters()) + list(self.q2.parameters()), lr=config.critic_lr)

        self.log_alpha = torch.tensor(0.0, device=self.device, requires_grad=True)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=config.alpha_lr)

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    def select_action(self, observation: np.ndarray, deterministic: bool = False) -> np.ndarray:
        observation_tensor = to_tensor(observation, self.device).unsqueeze(0)
        with torch.no_grad():
            if deterministic:
                action = self.actor.deterministic(observation_tensor)
            else:
                action, _, _ = self.actor.sample(observation_tensor)
        return action.squeeze(0).cpu().numpy()

    def update(self, batch: ReplayBatch) -> dict[str, float]:
        observations = to_tensor(batch.observations, self.device)
        actions = to_tensor(batch.actions, self.device)
        rewards = to_tensor(batch.rewards, self.device)
        next_observations = to_tensor(batch.next_observations, self.device)
        dones = to_tensor(batch.dones, self.device)

        with torch.no_grad():
            next_actions, next_log_prob, _ = self.actor.sample(next_observations)
            target_q = torch.min(
                self.target_q1(next_observations, next_actions),
                self.target_q2(next_observations, next_actions),
            )
            target_q = rewards + self.gamma * (1.0 - dones) * (target_q - self.alpha.detach() * next_log_prob)

        current_q1 = self.q1(observations, actions)
        current_q2 = self.q2(observations, actions)
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_optimizer.step()

        new_actions, log_prob, _ = self.actor.sample(observations)
        q1_new = self.q1(observations, new_actions)
        q2_new = self.q2(observations, new_actions)
        q_new = torch.min(q1_new, q2_new)
        actor_loss = (self.alpha.detach() * log_prob - q_new).mean()

        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_optimizer.step()

        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.alpha_optimizer.step()

        soft_update(self.target_q1, self.q1, self.tau)
        soft_update(self.target_q2, self.q2, self.tau)

        return {
            "critic_loss": float(critic_loss.item()),
            "actor_loss": float(actor_loss.item()),
            "alpha_loss": float(alpha_loss.item()),
            "alpha": float(self.alpha.item()),
            "log_prob": float(log_prob.mean().item()),
            "target_q": float(target_q.mean().item()),
        }

    def state_dict(self) -> dict[str, object]:
        return {
            "actor": self.actor.state_dict(),
            "q1": self.q1.state_dict(),
            "q2": self.q2.state_dict(),
            "target_q1": self.target_q1.state_dict(),
            "target_q2": self.target_q2.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu(),
            "alpha_optimizer": self.alpha_optimizer.state_dict(),
        }

    def load_state_dict(self, state_dict: dict[str, object]) -> None:
        self.actor.load_state_dict(state_dict["actor"])
        self.q1.load_state_dict(state_dict["q1"])
        self.q2.load_state_dict(state_dict["q2"])
        self.target_q1.load_state_dict(state_dict["target_q1"])
        self.target_q2.load_state_dict(state_dict["target_q2"])
        self.actor_optimizer.load_state_dict(state_dict["actor_optimizer"])
        self.critic_optimizer.load_state_dict(state_dict["critic_optimizer"])
        self.log_alpha = torch.tensor(float(state_dict["log_alpha"]), device=self.device, requires_grad=True)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=self.config.alpha_lr)
        self.alpha_optimizer.load_state_dict(state_dict["alpha_optimizer"])

