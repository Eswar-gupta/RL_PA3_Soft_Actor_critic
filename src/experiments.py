"""Experiment orchestration for the Reacher SAC runs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np

from src.train.train_sac import train_sac
from src.utils import ensure_dir, mean_and_confidence_interval, save_json


@dataclass(slots=True)
class ExperimentConfig:
    seeds: tuple[int, ...] = tuple(range(15))
    reward_formulations: tuple[str, ...] = ("Ra", "Rb", "Rc")
    total_steps: int = 1_000_000
    start_steps: int = 10_000
    eval_interval: int = 10_000
    eval_episodes: int = 20
    max_episode_steps: int = 1000


def run_reacher_experiments(
    *,
    config: ExperimentConfig | None = None,
    output_dir: str | Path = "experiments/reacher",
) -> dict[str, object]:
    """Train SAC-Ra, SAC-Rb and SAC-Rc across the requested seeds."""

    config = config or ExperimentConfig()
    output_path = ensure_dir(output_dir)

    all_results: list[dict[str, object]] = []
    for training_reward in config.reward_formulations:
        for seed in config.seeds:
            result = train_sac(
                training_reward,
                seed=seed,
                total_steps=config.total_steps,
                start_steps=config.start_steps,
                eval_interval=config.eval_interval,
                eval_episodes=config.eval_episodes,
                max_episode_steps=config.max_episode_steps,
                output_dir=output_path / training_reward.lower(),
                evaluation_reward_formulations=config.reward_formulations,
            )
            all_results.append(result)

    summary = summarize_results(all_results, config.reward_formulations)
    save_json(summary, output_path / "summary.json")
    return summary


def summarize_results(results: list[dict[str, object]], reward_formulations: Iterable[str]) -> dict[str, object]:
    """Aggregate returns across seeds for each training/evaluation pairing."""

    summary: dict[str, object] = {"runs": results, "aggregates": {}}
    for training_reward in reward_formulations:
        training_runs = [result for result in results if result["training_reward_formulation"] == training_reward]
        if not training_runs:
            continue

        evaluation_steps = training_runs[0]["evaluation_steps"]
        aggregate_by_eval: dict[str, dict[str, list[float]]] = {}

        for eval_reward in reward_formulations:
            per_step_values: list[list[float]] = []
            for run in training_runs:
                curves = run["evaluation_curves"]
                per_step_values.append(list(curves[eval_reward]))

            means: list[float] = []
            lower: list[float] = []
            upper: list[float] = []
            for step_index in range(len(evaluation_steps)):
                values = [series[step_index] for series in per_step_values]
                mean, ci_low, ci_high = mean_and_confidence_interval(values)
                means.append(mean)
                lower.append(ci_low)
                upper.append(ci_high)

            aggregate_by_eval[eval_reward] = {
                "mean": means,
                "ci_low": lower,
                "ci_high": upper,
            }

        summary["aggregates"][training_reward] = {
            "evaluation_steps": evaluation_steps,
            "by_reward": aggregate_by_eval,
        }

    return summary


def plot_reacher_summary(summary: dict[str, object], output_path: str | Path | None = None) -> None:
    """Plot mean and confidence intervals for each reward formulation."""

    aggregates = summary.get("aggregates", {})
    if not aggregates:
        return

    training_rewards = list(aggregates.keys())
    evaluation_rewards = list(next(iter(aggregates.values()))["by_reward"].keys())

    fig, axes = plt.subplots(len(training_rewards), 1, figsize=(10, 4 * len(training_rewards)), sharex=True)
    if len(training_rewards) == 1:
        axes = [axes]

    for axis, training_reward in zip(axes, training_rewards, strict=True):
        aggregate = aggregates[training_reward]
        steps = aggregate["evaluation_steps"]
        for evaluation_reward in evaluation_rewards:
            series = aggregate["by_reward"][evaluation_reward]
            mean = np.asarray(series["mean"], dtype=np.float64)
            ci_low = np.asarray(series["ci_low"], dtype=np.float64)
            ci_high = np.asarray(series["ci_high"], dtype=np.float64)
            axis.plot(steps, mean, label=f"Eval {evaluation_reward}")
            axis.fill_between(steps, ci_low, ci_high, alpha=0.2)
        axis.set_title(f"Training reward: {training_reward}")
        axis.set_ylabel("Average return")
        axis.legend()

    axes[-1].set_xlabel("Environment steps")
    fig.tight_layout()

    if output_path is not None:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_file, dpi=200, bbox_inches="tight")

    plt.close(fig)
