"""Replay buffer for off-policy continuous control."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class ReplayBatch:
    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_observations: np.ndarray
    dones: np.ndarray


class ReplayBuffer:
    """Fixed-size cyclic replay buffer."""

    def __init__(self, capacity: int, observation_dim: int, action_dim: int) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = int(capacity)
        self.observation_dim = int(observation_dim)
        self.action_dim = int(action_dim)

        self.observations = np.zeros((self.capacity, self.observation_dim), dtype=np.float32)
        self.actions = np.zeros((self.capacity, self.action_dim), dtype=np.float32)
        self.rewards = np.zeros((self.capacity, 1), dtype=np.float32)
        self.next_observations = np.zeros((self.capacity, self.observation_dim), dtype=np.float32)
        self.dones = np.zeros((self.capacity, 1), dtype=np.float32)

        self.position = 0
        self.size = 0

    def __len__(self) -> int:
        return self.size

    def add(
        self,
        observation: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_observation: np.ndarray,
        done: bool,
    ) -> None:
        index = self.position
        self.observations[index] = np.asarray(observation, dtype=np.float32).reshape(-1)
        self.actions[index] = np.asarray(action, dtype=np.float32).reshape(-1)
        self.rewards[index] = np.asarray(reward, dtype=np.float32).reshape(1)
        self.next_observations[index] = np.asarray(next_observation, dtype=np.float32).reshape(-1)
        self.dones[index] = np.asarray(done, dtype=np.float32).reshape(1)

        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> ReplayBatch:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.size == 0:
            raise ValueError("cannot sample from an empty replay buffer")

        indices = np.random.randint(0, self.size, size=batch_size)
        return ReplayBatch(
            observations=self.observations[indices],
            actions=self.actions[indices],
            rewards=self.rewards[indices],
            next_observations=self.next_observations[indices],
            dones=self.dones[indices],
        )
