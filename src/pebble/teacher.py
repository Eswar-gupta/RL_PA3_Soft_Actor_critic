"""Simulated teacher for generating preference labels from ground-truth rewards."""

from __future__ import annotations

from typing import Callable

import numpy as np

from src.pebble.preference_buffer import PreferencePair


class PreferenceTeacher:
	"""Teacher that generates preference labels based on ground-truth reward returns.
	
	The teacher compares two trajectory segments and returns a preference label based
	on their accumulated returns under a ground-truth reward function.
	"""

	def __init__(self, reward_fn: Callable[[np.ndarray, np.ndarray], float]) -> None:
		"""Initialize the teacher with a ground-truth reward function.
		
		Args:
			reward_fn: A callable that takes (observation, action) and returns a scalar reward.
				This is the ground-truth reward used to label preferences.
		"""
		self.reward_fn = reward_fn

	def get_preference_label(self, pair: PreferencePair) -> int:
		"""Get preference label for a pair based on ground-truth returns.
		
		Args:
			pair: A PreferencePair containing two segments and their returns.
			
		Returns:
			1 if segment 1 has higher return, 0 if segment 2 has higher or equal return.
		"""
		return 1 if pair.return_1 > pair.return_2 else 0

	def label_batch(self, pairs: list[PreferencePair]) -> np.ndarray:
		"""Generate preference labels for a batch of pairs.
		
		Args:
			pairs: List of PreferencePair objects.
			
		Returns:
			Array of preference labels (0 or 1) for each pair.
		"""
		labels = np.array([self.get_preference_label(pair) for pair in pairs], dtype=np.int32)
		return labels

	def get_segment_returns(self, pair: PreferencePair) -> tuple[float, float]:
		"""Get the returns for both segments in a pair.
		
		Args:
			pair: A PreferencePair object.
			
		Returns:
			Tuple of (return_1, return_2).
		"""
		return pair.return_1, pair.return_2
