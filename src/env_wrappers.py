"""Environment helpers and reward wrappers."""

from __future__ import annotations

import gymnasium as gym
import numpy as np

from src.rewards.pendulum_rewards import make_target_reward_function


REACHER_REWARD_FORMULATIONS = {"ra", "rb", "rc"}


def _normalise_formulation(formulation: str) -> str:
    key = formulation.strip().lower()
    if key not in REACHER_REWARD_FORMULATIONS:
        raise ValueError(f"Unknown Reacher reward formulation: {formulation!r}")
    return key


def _distance_from_observation(observation: np.ndarray, info: dict[str, object] | None = None) -> float:
    if info is not None and "reward_dist" in info:
        return float(-np.asarray(info["reward_dist"], dtype=np.float32))
    return float(np.linalg.norm(np.asarray(observation, dtype=np.float32).reshape(-1)[-2:], ord=2))


def _velocity_from_observation(observation: np.ndarray, info: dict[str, object] | None = None) -> np.ndarray:
    if info is not None and "qvel" in info:
        return np.asarray(info["qvel"], dtype=np.float32).reshape(-1)
    observation = np.asarray(observation, dtype=np.float32).reshape(-1)
    if observation.size >= 8:
        return observation[6:8]
    return np.zeros(2, dtype=np.float32)


class ReacherRewardWrapper(gym.Wrapper):
    """Replace the default Reacher reward with one of the assignment variants."""

    def __init__(
        self,
        env: gym.Env,
        reward_formulation: str,
        target_radius: float = 0.05,
        velocity_threshold: float = 1e-2,
    ) -> None:
        super().__init__(env)
        self.reward_formulation = _normalise_formulation(reward_formulation)
        self.target_radius = float(target_radius)
        self.velocity_threshold = float(velocity_threshold)

    def _in_target(self, observation: np.ndarray, info: dict[str, object]) -> bool:
        return _distance_from_observation(observation, info) <= self.target_radius

    def step(self, action):
        observation, _, terminated, truncated, info = self.env.step(action)
        info = dict(info)

        distance = _distance_from_observation(observation, info)
        velocity = _velocity_from_observation(observation, info)
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        in_target = distance <= self.target_radius

        if self.reward_formulation == "ra":
            if in_target:
                reward = 1.0
            else:
                reward_dist = float(info.get("reward_dist", -distance))
                reward_ctrl = float(info.get("reward_ctrl", -np.square(action).sum()))
                reward = reward_dist + reward_ctrl
        elif self.reward_formulation == "rb":
            reward = 1.0 if in_target else 0.0
        else:
            reward = -1.0
            terminated = bool(terminated or (in_target and float(np.linalg.norm(velocity, ord=2)) <= self.velocity_threshold))

        info.update(
            {
                "custom_reward": float(reward),
                "custom_reward_formulation": self.reward_formulation,
                "distance_to_target": float(distance),
                "velocity_norm": float(np.linalg.norm(velocity, ord=2)),
                "target_reached": bool(in_target),
            }
        )
        return observation, float(reward), bool(terminated), bool(truncated), info


class PendulumTargetRewardWrapper(gym.Wrapper):
    """Replace Pendulum reward with a target-angle reward."""

    def __init__(self, env: gym.Env, target_angle_deg: float) -> None:
        super().__init__(env)
        self.target_angle_deg = float(target_angle_deg)
        self.reward_fn = make_target_reward_function(self.target_angle_deg)

    def step(self, action):
        observation, _, terminated, truncated, info = self.env.step(action)
        info = dict(info)
        reward = self.reward_fn(observation, action)
        info.update(
            {
                "custom_reward": float(reward),
                "custom_reward_target_angle_deg": self.target_angle_deg,
            }
        )
        return observation, float(reward), bool(terminated), bool(truncated), info


def make_reacher_env(
    reward_formulation: str,
    *,
    max_episode_steps: int = 1000,
    target_radius: float = 0.05,
    velocity_threshold: float = 1e-2,
    seed: int | None = None,
    render_mode: str | None = None,
) -> gym.Env:
    """Construct a Reacher-v5 environment with the requested reward formulation."""

    env = gym.make(
        "Reacher-v5",
        max_episode_steps=max_episode_steps,
        render_mode=render_mode,
    )
    env = ReacherRewardWrapper(
        env,
        reward_formulation=reward_formulation,
        target_radius=target_radius,
        velocity_threshold=velocity_threshold,
    )
    env = gym.wrappers.RecordEpisodeStatistics(env)
    if seed is not None:
        env.reset(seed=seed)
    return env


def make_pendulum_env(
    target_angle_deg: float,
    *,
    max_episode_steps: int = 200,
    seed: int | None = None,
    render_mode: str | None = None,
) -> gym.Env:
    """Construct a Pendulum-v1 environment with a target-angle reward."""

    env = gym.make(
        "Pendulum-v1",
        max_episode_steps=max_episode_steps,
        render_mode=render_mode,
    )
    env = PendulumTargetRewardWrapper(env, target_angle_deg=target_angle_deg)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    if seed is not None:
        env.reset(seed=seed)
    return env

