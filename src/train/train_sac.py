"""Training and evaluation utilities for continuous SAC on Reacher."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from src.env_wrappers import make_reacher_env
from src.replay_buffer import ReplayBuffer
from src.sac.sac_agent import SACAgent, SACConfig
from src.utils import ensure_dir, save_json, set_seed


@dataclass(slots=True)
class EvaluationRecord:
    step: int
    reward_formulation: str
    mean_return: float
    episode_returns: list[float]


def evaluate_policy(
    agent: SACAgent,
    reward_formulation: str,
    *,
    episodes: int = 20,
    max_episode_steps: int = 1000,
    target_radius: float = 0.05,
    velocity_threshold: float = 1e-2,
    seed: int = 0,
) -> tuple[float, list[float]]:
    """Run offline deterministic evaluation episodes."""

    env = make_reacher_env(
        reward_formulation,
        max_episode_steps=max_episode_steps,
        target_radius=target_radius,
        velocity_threshold=velocity_threshold,
        seed=seed,
    )
    episode_returns: list[float] = []

    for episode_index in range(episodes):
        observation, _ = env.reset(seed=seed + episode_index)
        terminated = False
        truncated = False
        return_value = 0.0

        while not (terminated or truncated):
            action = agent.select_action(observation, deterministic=True)
            observation, reward, terminated, truncated, _ = env.step(action)
            return_value += float(reward)

        episode_returns.append(return_value)

    env.close()
    return float(np.mean(episode_returns)), episode_returns


def evaluate_policy_across_rewards(
    agent: SACAgent,
    reward_formulations: Iterable[str] = ("Ra", "Rb", "Rc"),
    *,
    episodes: int = 20,
    max_episode_steps: int = 1000,
    target_radius: float = 0.05,
    velocity_threshold: float = 1e-2,
    seed: int = 0,
) -> dict[str, tuple[float, list[float]]]:
    """Evaluate a policy under multiple Reacher reward formulations."""

    metrics: dict[str, tuple[float, list[float]]] = {}
    for offset, formulation in enumerate(reward_formulations):
        metrics[formulation] = evaluate_policy(
            agent,
            formulation,
            episodes=episodes,
            max_episode_steps=max_episode_steps,
            target_radius=target_radius,
            velocity_threshold=velocity_threshold,
            seed=seed + 1000 * offset,
        )
    return metrics


def train_sac(
    reward_formulation: str,
    *,
    seed: int = 0,
    total_steps: int = 1_000_000,
    start_steps: int = 10_000,
    batch_size: int = 256,
    buffer_capacity: int = 1_000_000,
    eval_interval: int = 10_000,
    eval_episodes: int = 20,
    max_episode_steps: int = 1000,
    target_radius: float = 0.05,
    velocity_threshold: float = 1e-2,
    gamma: float = 0.99,
    tau: float = 0.005,
    actor_lr: float = 3e-4,
    critic_lr: float = 3e-4,
    alpha_lr: float = 3e-4,
    hidden_sizes: tuple[int, ...] = (256, 256),
    target_entropy: float | None = None,
    device: str = "cpu",
    output_dir: str | Path | None = None,
    evaluation_reward_formulations: Iterable[str] = ("Ra", "Rb", "Rc"),
) -> dict[str, object]:
    """Train SAC on one reward formulation and evaluate on all three."""

    set_seed(seed)
    env = make_reacher_env(
        reward_formulation,
        max_episode_steps=max_episode_steps,
        target_radius=target_radius,
        velocity_threshold=velocity_threshold,
        seed=seed,
    )

    observation_dim = int(env.observation_space.shape[0])
    action_space = env.action_space
    action_dim = int(np.prod(action_space.shape))

    agent = SACAgent(
        SACConfig(
            observation_dim=observation_dim,
            action_dim=action_dim,
            action_low=action_space.low,
            action_high=action_space.high,
            hidden_sizes=hidden_sizes,
            gamma=gamma,
            tau=tau,
            actor_lr=actor_lr,
            critic_lr=critic_lr,
            alpha_lr=alpha_lr,
            target_entropy=target_entropy,
            device=device,
        )
    )
    buffer = ReplayBuffer(buffer_capacity, observation_dim, action_dim)

    observation, _ = env.reset(seed=seed)
    evaluation_steps: list[int] = []
    evaluation_records: list[EvaluationRecord] = []
    evaluation_curves: dict[str, list[float]] = {formulation: [] for formulation in evaluation_reward_formulations}

    for step in range(1, total_steps + 1):
        if step <= start_steps:
            action = action_space.sample()
        else:
            action = agent.select_action(observation, deterministic=False)

        next_observation, reward, terminated, truncated, _ = env.step(action)
        done = bool(terminated or truncated)
        buffer.add(observation, action, reward, next_observation, done)

        observation = next_observation

        if step > start_steps and len(buffer) >= batch_size:
            agent.update(buffer.sample(batch_size))

        if done:
            observation, _ = env.reset()

        if step % eval_interval == 0:
            evaluation_steps.append(step)
            metrics = evaluate_policy_across_rewards(
                agent,
                reward_formulations=evaluation_reward_formulations,
                episodes=eval_episodes,
                max_episode_steps=max_episode_steps,
                target_radius=target_radius,
                velocity_threshold=velocity_threshold,
                seed=seed + step,
            )
            for formulation, (mean_return, episode_returns) in metrics.items():
                evaluation_curves[formulation].append(mean_return)
                evaluation_records.append(
                    EvaluationRecord(
                        step=step,
                        reward_formulation=formulation,
                        mean_return=mean_return,
                        episode_returns=episode_returns,
                    )
                )

    env.close()

    result = {
        "training_reward_formulation": reward_formulation,
        "seed": seed,
        "evaluation_steps": evaluation_steps,
        "evaluation_curves": evaluation_curves,
        "evaluation_records": [asdict(record) for record in evaluation_records],
    }

    if output_dir is not None:
        output_path = ensure_dir(output_dir)
        save_json(result, output_path / f"sac_{reward_formulation.lower()}_seed_{seed}.json")

    return result


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train SAC on Reacher reward formulations.")
    parser.add_argument("--reward-formulation", default="Ra", choices=["Ra", "Rb", "Rc", "ra", "rb", "rc"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--total-steps", type=int, default=1_000_000)
    parser.add_argument("--start-steps", type=int, default=10_000)
    parser.add_argument("--eval-interval", type=int, default=10_000)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--max-episode-steps", type=int, default=1000)
    parser.add_argument("--output-dir", type=str, default="experiments/reacher")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    train_sac(
        args.reward_formulation,
        seed=args.seed,
        total_steps=args.total_steps,
        start_steps=args.start_steps,
        eval_interval=args.eval_interval,
        eval_episodes=args.eval_episodes,
        max_episode_steps=args.max_episode_steps,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
