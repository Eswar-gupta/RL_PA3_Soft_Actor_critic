"""Reward formulations for the Reacher task.

The assignment figure defines three variants, named here as Ra, Rb and Rc.

Ra:
	r_t = 1 if the end-effector is in the target region, otherwise
	r_t = -||x_goal - x_pos|| - ||action||^2

Rb:
	r_t = 1 if the end-effector is in the target region, otherwise 0

Rc:
	r_t = -1 at every step until termination.

The termination rule for Rc is handled separately so the training loop can
decide when to end or resume an episode.
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np

ArrayLike = Sequence[float] | np.ndarray


def _l2_norm(vector: ArrayLike) -> float:
	"""Return the Euclidean norm of a vector as a plain float."""

	return float(np.linalg.norm(np.asarray(vector, dtype=np.float32)))


def _squared_l2_norm(vector: ArrayLike) -> float:
	"""Return the squared Euclidean norm of a vector as a plain float."""

	values = np.asarray(vector, dtype=np.float32)
	return float(np.sum(values * values))


def is_in_target(
	goal_position: ArrayLike,
	end_effector_position: ArrayLike,
	target_radius: float = 0.05,
) -> bool:
	"""Check whether the end-effector is inside the target region."""

	distance = _l2_norm(np.asarray(goal_position) - np.asarray(end_effector_position))
	return distance <= target_radius


def has_near_zero_velocity(velocity: ArrayLike, velocity_threshold: float = 1e-2) -> bool:
	"""Check whether the state velocity is small enough to count as stationary."""

	return _l2_norm(velocity) <= velocity_threshold


def reward_ra(
	goal_position: ArrayLike,
	end_effector_position: ArrayLike,
	action: ArrayLike,
	target_radius: float = 0.05,
) -> float:
	"""Compute Ra from the assignment figure."""

	if is_in_target(goal_position, end_effector_position, target_radius=target_radius):
		return 1.0

	distance_penalty = _l2_norm(np.asarray(goal_position) - np.asarray(end_effector_position))
	action_penalty = _squared_l2_norm(action)
	return -(distance_penalty + action_penalty)


def reward_rb(
	goal_position: ArrayLike,
	end_effector_position: ArrayLike,
	target_radius: float = 0.05,
) -> float:
	"""Compute Rb from the assignment figure."""

	return 1.0 if is_in_target(goal_position, end_effector_position, target_radius=target_radius) else 0.0


def reward_rc(*_: object, **__: object) -> float:
	"""Compute Rc from the assignment figure."""

	return -1.0


def rc_terminated(
	goal_position: ArrayLike,
	end_effector_position: ArrayLike,
	velocity: ArrayLike,
	target_radius: float = 0.05,
	velocity_threshold: float = 1e-2,
) -> bool:
	"""Return True when the Rc episode should terminate."""

	return is_in_target(goal_position, end_effector_position, target_radius=target_radius) and has_near_zero_velocity(
		velocity, velocity_threshold=velocity_threshold
	)


def get_reward_function(formulation: str) -> Callable[..., float]:
	"""Return the requested Reacher reward function.

	Parameters
	----------
	formulation:
		One of ``"Ra"``, ``"Rb"`` or ``"Rc"`` (case-insensitive).
	"""

	key = formulation.strip().lower()
	if key in {"ra", "reward_a", "a"}:
		return reward_ra
	if key in {"rb", "reward_b", "b"}:
		return reward_rb
	if key in {"rc", "reward_c", "c"}:
		return reward_rc
	raise ValueError(f"Unknown Reacher reward formulation: {formulation!r}")


__all__ = [
	"get_reward_function",
	"has_near_zero_velocity",
	"is_in_target",
	"rc_terminated",
	"reward_ra",
	"reward_rb",
	"reward_rc",
]
