"""Preference buffer for storing trajectory segments and sampling preference pairs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class TrajectorySegment:
	"""A segment of a trajectory used for preference learning."""

	observation: np.ndarray
	action: np.ndarray
	reward: float
	next_observation: np.ndarray
	done: bool
	trajectory_id: int
	step_in_trajectory: int


@dataclass(slots=True)
class PreferencePair:
	"""A pair of trajectory segments for preference labeling."""

	segment_1: TrajectorySegment
	segment_2: TrajectorySegment
	return_1: float
	return_2: float


class PreferenceBuffer:
	"""Buffer for storing trajectory segments and sampling preference pairs.
	
	This buffer accumulates trajectory segments and supports:
	- Pair sampling: returns two segments and their accumulated returns
	- Relabeling: replaces stored rewards with learned rewards from a reward model
	"""

	def __init__(self, capacity: int) -> None:
		"""Initialize the preference buffer.
		
		Args:
			capacity: Maximum number of segments to store.
		"""
		if capacity <= 0:
			raise ValueError("capacity must be positive")
		self.capacity = int(capacity)
		self.segments: list[TrajectorySegment] = []
		self.trajectory_returns: dict[int, list[float]] = {}  # trajectory_id -> returns for each step
		self.trajectory_step_counts: dict[int, int] = {}  # trajectory_id -> number of steps
		self.next_trajectory_id = 0
		self.segment_position = 0  # For cyclic insertion

	def __len__(self) -> int:
		"""Return the number of segments stored."""
		return len(self.segments)

	def add_step(
		self,
		observation: np.ndarray,
		action: np.ndarray,
		reward: float,
		next_observation: np.ndarray,
		done: bool,
	) -> None:
		"""Add a single step to the current trajectory.
		
		Args:
			observation: State observation.
			action: Action taken.
			reward: Reward received.
			next_observation: Next state observation.
			done: Whether the episode terminated.
		"""
		traj_id = self.next_trajectory_id

		# Initialize trajectory tracking if needed
		if traj_id not in self.trajectory_returns:
			self.trajectory_returns[traj_id] = []
			self.trajectory_step_counts[traj_id] = 0

		step_idx = self.trajectory_step_counts[traj_id]

		# Create segment
		segment = TrajectorySegment(
			observation=np.asarray(observation, dtype=np.float32).copy(),
			action=np.asarray(action, dtype=np.float32).copy(),
			reward=float(reward),
			next_observation=np.asarray(next_observation, dtype=np.float32).copy(),
			done=bool(done),
			trajectory_id=traj_id,
			step_in_trajectory=step_idx,
		)

		# Add to buffer (cyclic)
		if len(self.segments) < self.capacity:
			self.segments.append(segment)
		else:
			self.segments[self.segment_position] = segment
			self.segment_position = (self.segment_position + 1) % self.capacity

		# Track reward for this step
		self.trajectory_returns[traj_id].append(float(reward))
		self.trajectory_step_counts[traj_id] += 1

		# If episode is done, start a new trajectory
		if done:
			self.next_trajectory_id += 1

	def get_return_for_segment(self, segment: TrajectorySegment, gamma: float = 0.99) -> float:
		"""Compute the cumulative discounted return starting from a segment.
		
		Args:
			segment: The trajectory segment.
			gamma: Discount factor.
			
		Returns:
			Discounted cumulative return from this segment to episode end.
		"""
		traj_id = segment.trajectory_id
		step_idx = segment.step_in_trajectory

		if traj_id not in self.trajectory_returns:
			return float(segment.reward)

		rewards = self.trajectory_returns[traj_id]
		if step_idx >= len(rewards):
			return 0.0

		# Compute return: sum of discounted future rewards from this step onward
		ret = 0.0
		for i, r in enumerate(rewards[step_idx:]):
			ret += (gamma ** i) * r

		return float(ret)

	def sample_preference_pair(self, gamma: float = 0.99) -> PreferencePair | None:
		"""Sample a pair of segments from different trajectories.
		
		Args:
			gamma: Discount factor for computing returns.
			
		Returns:
			A preference pair, or None if not enough segments available.
		"""
		if len(self.segments) < 2:
			return None

		# Sample two segments
		indices = np.random.choice(len(self.segments), size=2, replace=False)
		seg1 = self.segments[indices[0]]
		seg2 = self.segments[indices[1]]

		# If from same trajectory, resample seg2
		attempts = 0
		while seg1.trajectory_id == seg2.trajectory_id and attempts < 10:
			seg2_idx = np.random.randint(0, len(self.segments))
			seg2 = self.segments[seg2_idx]
			attempts += 1

		# Compute returns
		ret1 = self.get_return_for_segment(seg1, gamma=gamma)
		ret2 = self.get_return_for_segment(seg2, gamma=gamma)

		return PreferencePair(
			segment_1=seg1,
			segment_2=seg2,
			return_1=ret1,
			return_2=ret2,
		)

	def sample_batch_preference_pairs(
		self, batch_size: int, gamma: float = 0.99
	) -> list[PreferencePair]:
		"""Sample a batch of preference pairs.
		
		Args:
			batch_size: Number of pairs to sample.
			gamma: Discount factor.
			
		Returns:
			List of preference pairs.
		"""
		pairs = []
		for _ in range(batch_size):
			pair = self.sample_preference_pair(gamma=gamma)
			if pair is not None:
				pairs.append(pair)
		return pairs

	def relabel_with_learned_rewards(
		self, learned_rewards_dict: dict[int, np.ndarray]
	) -> None:
		"""Relabel stored segments with learned rewards from a reward model.
		
		Args:
			learned_rewards_dict: Dictionary mapping segment index to learned reward.
		"""
		for idx, segment in enumerate(self.segments):
			if idx in learned_rewards_dict:
				segment.reward = float(learned_rewards_dict[idx])
