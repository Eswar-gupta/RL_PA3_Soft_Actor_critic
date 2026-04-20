"""PEBBLE policy-side adapter built on top of SAC.

This module intentionally keeps policy optimization thin and delegates all actor-critic
logic to the existing SAC implementation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.replay_buffer import ReplayBatch
from src.sac.sac_agent import SACAgent, SACConfig


@dataclass(slots=True)
class PebbleSACConfig:
	"""Configuration for SAC when used as the policy optimizer in PEBBLE."""

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
	device: str = "cpu"

	def to_sac_config(self) -> SACConfig:
		return SACConfig(
			observation_dim=self.observation_dim,
			action_dim=self.action_dim,
			action_low=self.action_low,
			action_high=self.action_high,
			hidden_sizes=self.hidden_sizes,
			gamma=self.gamma,
			tau=self.tau,
			actor_lr=self.actor_lr,
			critic_lr=self.critic_lr,
			alpha_lr=self.alpha_lr,
			target_entropy=self.target_entropy,
			device=self.device,
		)


class PebbleAgent:
	"""Policy optimizer for PEBBLE that reuses the SAC implementation."""

	def __init__(self, config: PebbleSACConfig) -> None:
		self.config = config
		self.sac = SACAgent(config.to_sac_config())

	@property
	def alpha(self) -> float:
		"""Current entropy temperature."""

		return float(self.sac.alpha.item())

	def select_action(self, observation: np.ndarray, deterministic: bool = False) -> np.ndarray:
		"""Select an action using the SAC policy."""

		return self.sac.select_action(observation, deterministic=deterministic)

	def update_from_batch(self, batch: ReplayBatch) -> dict[str, float]:
		"""Run one SAC update using a batch that already contains learned rewards."""

		return self.sac.update(batch)

	def update_with_learned_rewards(
		self,
		batch: ReplayBatch,
		learned_rewards: np.ndarray,
	) -> dict[str, float]:
		"""Run one SAC update after replacing batch rewards with learned rewards."""

		rewards = np.asarray(learned_rewards, dtype=np.float32).reshape(-1, 1)
		if rewards.shape[0] != batch.rewards.shape[0]:
			raise ValueError("learned_rewards batch size does not match replay batch")

		updated_batch = ReplayBatch(
			observations=batch.observations,
			actions=batch.actions,
			rewards=rewards,
			next_observations=batch.next_observations,
			dones=batch.dones,
		)
		return self.sac.update(updated_batch)

	def update_from_arrays(
		self,
		observations: np.ndarray,
		actions: np.ndarray,
		learned_rewards: np.ndarray,
		next_observations: np.ndarray,
		dones: np.ndarray,
	) -> dict[str, float]:
		"""Convenience helper to update directly from numpy arrays."""

		batch = ReplayBatch(
			observations=np.asarray(observations, dtype=np.float32),
			actions=np.asarray(actions, dtype=np.float32),
			rewards=np.asarray(learned_rewards, dtype=np.float32).reshape(-1, 1),
			next_observations=np.asarray(next_observations, dtype=np.float32),
			dones=np.asarray(dones, dtype=np.float32).reshape(-1, 1),
		)
		return self.sac.update(batch)

	def state_dict(self) -> dict[str, object]:
		"""Return the underlying SAC checkpoint data."""

		return self.sac.state_dict()

	def load_state_dict(self, state_dict: dict[str, object]) -> None:
		"""Load the underlying SAC checkpoint data."""

		self.sac.load_state_dict(state_dict)

