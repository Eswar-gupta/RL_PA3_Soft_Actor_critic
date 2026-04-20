"""Pendulum reward utilities for target-angle tasks."""

from __future__ import annotations

import math

import numpy as np


PENDULUM_TARGET_ANGLES_DEG: tuple[int, ...] = (0, -60, 90, 120, -150)


def angle_normalize(angle: float) -> float:
	"""Normalize angle to the interval [-pi, pi]."""

	return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


def parse_pendulum_observation(observation: np.ndarray) -> tuple[float, float]:
	"""Extract (theta, theta_dot) from Pendulum observation [cos(theta), sin(theta), theta_dot]."""

	obs = np.asarray(observation, dtype=np.float32).reshape(-1)
	if obs.size < 3:
		raise ValueError("Pendulum observation must have at least 3 elements")
	theta = float(np.arctan2(obs[1], obs[0]))
	theta_dot = float(obs[2])
	return theta, theta_dot


def pendulum_target_reward(
	observation: np.ndarray,
	action: np.ndarray | float,
	*,
	target_angle_rad: float,
	theta_dot_weight: float = 0.1,
	action_weight: float = 0.001,
) -> float:
	"""Compute the target-angle Pendulum reward used for this assignment.

	Reward is the negative cost:
		-(angle_error^2 + 0.1 * theta_dot^2 + 0.001 * torque^2)
	where angle_error = angle_normalize(theta - theta_target).
	"""

	theta, theta_dot = parse_pendulum_observation(observation)
	torque = float(np.asarray(action, dtype=np.float32).reshape(-1)[0])
	angle_error = angle_normalize(theta - float(target_angle_rad))
	cost = angle_error * angle_error + theta_dot_weight * (theta_dot * theta_dot) + action_weight * (torque * torque)
	return float(-cost)


def make_target_reward_function(
	target_angle_deg: float,
	*,
	theta_dot_weight: float = 0.1,
	action_weight: float = 0.001,
):
	"""Return a callable reward(obs, action) for a given target angle in degrees."""

	target_angle_rad = math.radians(float(target_angle_deg))

	def reward_fn(observation: np.ndarray, action: np.ndarray | float) -> float:
		return pendulum_target_reward(
			observation,
			action,
			target_angle_rad=target_angle_rad,
			theta_dot_weight=theta_dot_weight,
			action_weight=action_weight,
		)

	return reward_fn


def is_supported_target_angle(target_angle_deg: float) -> bool:
	"""Check whether the angle is one of the assignment targets."""

	return int(round(float(target_angle_deg))) in set(PENDULUM_TARGET_ANGLES_DEG)
