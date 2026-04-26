"""Step 5 experiments: Pendulum SAC baseline vs PEBBLE with simulated teacher."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import NormalDist

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from src.utils import ensure_dir, save_json, set_seed
from src.env_wrappers import make_pendulum_env
from src.replay_buffer import ReplayBuffer
from src.rewards.pendulum_rewards import PENDULUM_TARGET_ANGLES_DEG, make_target_reward_function
from src.sac.sac_agent import SACAgent, SACConfig


from src.pebble import PebbleAgent, PebbleSACConfig, PreferenceBuffer, PreferenceTeacher


def _default_device() -> str:
    """Prefer CUDA when available."""

    return "cuda" if torch.cuda.is_available() else "cpu"


@dataclass(slots=True)
class Step5Config:
    seeds: tuple[int, ...] = tuple(range(15))
    target_angles_deg: tuple[int, ...] = PENDULUM_TARGET_ANGLES_DEG
    preference_budgets: tuple[int, ...] = (500, 1000, 3000)
    total_steps: int = 300_000
    start_steps: int = 10_000
    eval_interval: int = 10_000
    eval_episodes: int = 20
    max_episode_steps: int = 200
    batch_size: int = 256
    buffer_capacity: int = 1_000_000
    gamma: float = 0.99
    tau: float = 0.005
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4
    hidden_sizes: tuple[int, ...] = (256, 256)
    target_entropy: float | None = None
    device: str = _default_device()
    query_interval: int = 2_000
    query_batch_size: int = 20
    reward_model_epochs: int = 5
    reward_model_batch_size: int = 64


@dataclass(slots=True)
class RunResult:
    method: str
    target_angle_deg: int
    seed: int
    preference_budget: int | None
    evaluation_steps: list[int]
    mean_returns: list[float]
    preference_queries_used: int


class PairwiseLinearRewardModel(nn.Module):
    """Simple reward model trained with Bradley-Terry pairwise comparisons."""

    def __init__(self, observation_dim: int, action_dim: int, device: str = "cpu") -> None:
        super().__init__()
        self.device = torch.device(device)
        feature_dim = observation_dim + action_dim
        self.linear = nn.Linear(feature_dim, 1, bias=True)
        self.to(self.device)

    def reward(self, observations: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        features = torch.cat([observations, actions], dim=1)
        return self.linear(features)

    def predict_reward(self, observations: np.ndarray, actions: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            obs = torch.as_tensor(observations, dtype=torch.float32, device=self.device)
            act = torch.as_tensor(actions, dtype=torch.float32, device=self.device)
            rewards = self.reward(obs, act)
            return rewards.cpu().numpy().reshape(-1, 1)

    def train_on_pairs(
        self,
        pair_features_1: np.ndarray,
        pair_features_2: np.ndarray,
        labels: np.ndarray,
        *,
        epochs: int,
        batch_size: int,
        learning_rate: float = 1e-3,
    ) -> list[float]:
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        features_1 = torch.as_tensor(pair_features_1, dtype=torch.float32, device=self.device)
        features_2 = torch.as_tensor(pair_features_2, dtype=torch.float32, device=self.device)
        labels_tensor = torch.as_tensor(labels, dtype=torch.float32, device=self.device).reshape(-1, 1)

        losses: list[float] = []
        n_samples = int(features_1.shape[0])
        if n_samples == 0:
            return losses

        for _ in range(epochs):
            permutation = torch.randperm(n_samples, device=self.device)
            epoch_loss = 0.0
            n_batches = 0
            for start in range(0, n_samples, batch_size):
                idx = permutation[start : start + batch_size]
                f1 = features_1[idx]
                f2 = features_2[idx]
                lbl = labels_tensor[idx]

                logits = self.linear(f1) - self.linear(f2)
                loss = F.binary_cross_entropy_with_logits(logits, lbl)

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

                epoch_loss += float(loss.item())
                n_batches += 1

            losses.append(epoch_loss / max(n_batches, 1))
        return losses


def _z_critical(confidence: float = 0.95) -> float:
    return float(NormalDist().inv_cdf((1.0 + confidence) / 2.0))


def _mean_ci(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    means = np.mean(values, axis=0)
    if values.shape[0] <= 1:
        return means, means, means
    std = np.std(values, axis=0, ddof=1)
    sem = std / np.sqrt(values.shape[0])
    z = _z_critical()
    delta = z * sem
    return means, means - delta, means + delta


def evaluate_pendulum_policy(
    agent: SACAgent | PebbleAgent,
    target_angle_deg: int,
    *,
    episodes: int,
    max_episode_steps: int,
    seed: int,
) -> float:
    env = make_pendulum_env(
        target_angle_deg,
        max_episode_steps=max_episode_steps,
        seed=seed,
    )
    returns: list[float] = []

    for episode in range(episodes):
        observation, _ = env.reset(seed=seed + episode)
        terminated = False
        truncated = False
        episode_return = 0.0
        while not (terminated or truncated):
            action = agent.select_action(observation, deterministic=True)
            observation, reward, terminated, truncated, _ = env.step(action)
            episode_return += float(reward)
        returns.append(episode_return)

    env.close()
    return float(np.mean(returns))


def _pair_features_from_preferences(pairs) -> tuple[np.ndarray, np.ndarray]:
    features_1 = []
    features_2 = []
    for pair in pairs:
        s1 = np.concatenate([pair.segment_1.observation, pair.segment_1.action], axis=0)
        s2 = np.concatenate([pair.segment_2.observation, pair.segment_2.action], axis=0)
        features_1.append(s1)
        features_2.append(s2)
    return np.asarray(features_1, dtype=np.float32), np.asarray(features_2, dtype=np.float32)


def train_sac_pendulum_ground_truth(
    target_angle_deg: int,
    *,
    config: Step5Config,
    seed: int,
) -> RunResult:
    set_seed(seed)
    env = make_pendulum_env(
        target_angle_deg,
        max_episode_steps=config.max_episode_steps,
        seed=seed,
    )

    observation_dim = int(env.observation_space.shape[0])
    action_dim = int(np.prod(env.action_space.shape))
    action_space = env.action_space

    agent = SACAgent(
        SACConfig(
            observation_dim=observation_dim,
            action_dim=action_dim,
            action_low=action_space.low,
            action_high=action_space.high,
            hidden_sizes=config.hidden_sizes,
            gamma=config.gamma,
            tau=config.tau,
            actor_lr=config.actor_lr,
            critic_lr=config.critic_lr,
            alpha_lr=config.alpha_lr,
            target_entropy=config.target_entropy,
            device=config.device,
        )
    )
    replay_buffer = ReplayBuffer(config.buffer_capacity, observation_dim, action_dim)

    observation, _ = env.reset(seed=seed)
    evaluation_steps: list[int] = []
    mean_returns: list[float] = []

    for step in range(1, config.total_steps + 1):
        if step <= config.start_steps:
            action = action_space.sample()
        else:
            action = agent.select_action(observation, deterministic=False)

        next_observation, reward, terminated, truncated, _ = env.step(action)
        done = bool(terminated or truncated)
        replay_buffer.add(observation, action, reward, next_observation, done)
        observation = next_observation

        if step > config.start_steps and len(replay_buffer) >= config.batch_size:
            batch = replay_buffer.sample(config.batch_size)
            agent.update(batch)

        if done:
            observation, _ = env.reset()

        if step % config.eval_interval == 0:
            evaluation_steps.append(step)
            mean_ret = evaluate_pendulum_policy(
                agent,
                target_angle_deg,
                episodes=config.eval_episodes,
                max_episode_steps=config.max_episode_steps,
                seed=seed + step,
            )
            mean_returns.append(mean_ret)

    env.close()
    return RunResult(
        method="sac_ground_truth",
        target_angle_deg=target_angle_deg,
        seed=seed,
        preference_budget=None,
        evaluation_steps=evaluation_steps,
        mean_returns=mean_returns,
        preference_queries_used=0,
    )


def train_pebble_pendulum(
    target_angle_deg: int,
    preference_budget: int,
    *,
    config: Step5Config,
    seed: int,
) -> RunResult:
    set_seed(seed)
    env = make_pendulum_env(
        target_angle_deg,
        max_episode_steps=config.max_episode_steps,
        seed=seed,
    )

    observation_dim = int(env.observation_space.shape[0])
    action_dim = int(np.prod(env.action_space.shape))
    action_space = env.action_space

    agent = PebbleAgent(
        PebbleSACConfig(
            observation_dim=observation_dim,
            action_dim=action_dim,
            action_low=action_space.low,
            action_high=action_space.high,
            hidden_sizes=config.hidden_sizes,
            gamma=config.gamma,
            tau=config.tau,
            actor_lr=config.actor_lr,
            critic_lr=config.critic_lr,
            alpha_lr=config.alpha_lr,
            target_entropy=config.target_entropy,
            device=config.device,
        )
    )

    replay_buffer = ReplayBuffer(config.buffer_capacity, observation_dim, action_dim)
    preference_buffer = PreferenceBuffer(capacity=config.buffer_capacity)
    teacher_fn = make_target_reward_function(target_angle_deg)
    teacher = PreferenceTeacher(teacher_fn)
    reward_model = PairwiseLinearRewardModel(observation_dim, action_dim, device=config.device)

    preference_queries_used = 0

    observation, _ = env.reset(seed=seed)
    evaluation_steps: list[int] = []
    mean_returns: list[float] = []

    for step in range(1, config.total_steps + 1):
        if step <= config.start_steps:
            action = action_space.sample()
        else:
            action = agent.select_action(observation, deterministic=False)

        next_observation, ground_truth_reward, terminated, truncated, _ = env.step(action)
        done = bool(terminated or truncated)

        replay_buffer.add(observation, action, ground_truth_reward, next_observation, done)
        preference_buffer.add_step(observation, action, ground_truth_reward, next_observation, done)
        observation = next_observation

        can_query = (
            step % config.query_interval == 0
            and len(preference_buffer) >= 2 * config.query_batch_size
            and preference_queries_used < preference_budget
        )
        if can_query:
            remaining = preference_budget - preference_queries_used
            query_size = int(min(config.query_batch_size, remaining))
            pairs = preference_buffer.sample_batch_preference_pairs(query_size, gamma=config.gamma)
            if pairs:
                labels = teacher.label_batch(pairs)
                f1, f2 = _pair_features_from_preferences(pairs)
                reward_model.train_on_pairs(
                    f1,
                    f2,
                    labels,
                    epochs=config.reward_model_epochs,
                    batch_size=config.reward_model_batch_size,
                )
                preference_queries_used += len(pairs)

        if step > config.start_steps and len(replay_buffer) >= config.batch_size:
            batch = replay_buffer.sample(config.batch_size)
            learned_rewards = reward_model.predict_reward(batch.observations, batch.actions)
            agent.update_with_learned_rewards(batch, learned_rewards)

        if done:
            observation, _ = env.reset()

        if step % config.eval_interval == 0:
            evaluation_steps.append(step)
            mean_ret = evaluate_pendulum_policy(
                agent,
                target_angle_deg,
                episodes=config.eval_episodes,
                max_episode_steps=config.max_episode_steps,
                seed=seed + step,
            )
            mean_returns.append(mean_ret)

    env.close()
    return RunResult(
        method="pebble",
        target_angle_deg=target_angle_deg,
        seed=seed,
        preference_budget=preference_budget,
        evaluation_steps=evaluation_steps,
        mean_returns=mean_returns,
        preference_queries_used=preference_queries_used,
    )


def _aggregate_method_results(results: list[RunResult]) -> dict[str, object]:
    if not results:
        return {}
    steps = results[0].evaluation_steps
    curves = np.asarray([run.mean_returns for run in results], dtype=np.float64)
    mean, low, high = _mean_ci(curves)

    auc_values = [float(np.trapz(np.asarray(run.mean_returns, dtype=np.float64), x=np.asarray(steps, dtype=np.float64))) for run in results]
    final_values = [float(np.mean(run.mean_returns[-10:])) if run.mean_returns else 0.0 for run in results]

    auc_mean, auc_low, auc_high = _mean_ci(np.asarray(auc_values, dtype=np.float64).reshape(-1, 1))
    final_mean, final_low, final_high = _mean_ci(np.asarray(final_values, dtype=np.float64).reshape(-1, 1))

    return {
        "evaluation_steps": steps,
        "mean_curve": mean.tolist(),
        "ci_low": low.tolist(),
        "ci_high": high.tolist(),
        "learning_efficiency_auc": {
            "mean": float(auc_mean[0]),
            "ci_low": float(auc_low[0]),
            "ci_high": float(auc_high[0]),
        },
        "final_performance_last10": {
            "mean": float(final_mean[0]),
            "ci_low": float(final_low[0]),
            "ci_high": float(final_high[0]),
        },
        "num_runs": len(results),
    }


def _plot_target_comparison(
    target_angle_deg: int,
    aggregates: dict[str, dict[str, object]],
    output_file: Path,
) -> None:
    plt.figure(figsize=(10, 6))

    for label, series in aggregates.items():
        steps = np.asarray(series["evaluation_steps"], dtype=np.float64)
        mean = np.asarray(series["mean_curve"], dtype=np.float64)
        ci_low = np.asarray(series["ci_low"], dtype=np.float64)
        ci_high = np.asarray(series["ci_high"], dtype=np.float64)
        plt.plot(steps, mean, label=label)
        plt.fill_between(steps, ci_low, ci_high, alpha=0.2)

    plt.title(f"Pendulum Target Angle {target_angle_deg} deg")
    plt.xlabel("Environment timesteps")
    plt.ylabel("Average undiscounted ground-truth return")
    plt.legend()
    plt.tight_layout()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=200, bbox_inches="tight")
    plt.close()


def _plot_normalized_summary(summary: dict[int, dict[str, dict[str, object]]], output_file: Path) -> None:
    # Normalize each target by SAC final mean to avoid scale mismatch across targets.
    method_to_curves: dict[str, list[np.ndarray]] = {}
    steps_ref: np.ndarray | None = None

    for target_data in summary.values():
        sac_key = "SAC (ground truth)"
        if sac_key not in target_data:
            continue
        sac_curve = np.asarray(target_data[sac_key]["mean_curve"], dtype=np.float64)
        scale = float(abs(sac_curve[-1])) if sac_curve.size > 0 else 1.0
        if scale < 1e-6:
            scale = 1.0

        for label, data in target_data.items():
            curve = np.asarray(data["mean_curve"], dtype=np.float64) / scale
            method_to_curves.setdefault(label, []).append(curve)
            if steps_ref is None:
                steps_ref = np.asarray(data["evaluation_steps"], dtype=np.float64)

    if steps_ref is None:
        return

    plt.figure(figsize=(10, 6))
    for label, curves in method_to_curves.items():
        stacked = np.stack(curves, axis=0)
        mean, low, high = _mean_ci(stacked)
        plt.plot(steps_ref, mean, label=label)
        plt.fill_between(steps_ref, low, high, alpha=0.2)

    plt.title("Step 5 Summary (normalized across target angles)")
    plt.xlabel("Environment timesteps")
    plt.ylabel("Normalized average undiscounted return")
    plt.legend()
    plt.tight_layout()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=200, bbox_inches="tight")
    plt.close()


def run_step5_experiments(
    *,
    config: Step5Config | None = None,
    output_dir: str | Path = "src/pebble/results/step5_pendulum",
) -> dict[str, object]:
    config = config or Step5Config()
    root = ensure_dir(output_dir)

    all_runs: list[RunResult] = []
    aggregated_by_target: dict[int, dict[str, dict[str, object]]] = {}

    for target in config.target_angles_deg:
        target_runs: dict[str, list[RunResult]] = {"SAC (ground truth)": []}
        for budget in config.preference_budgets:
            target_runs[f"PEBBLE (budget={budget})"] = []

        for seed in config.seeds:
            sac_run = train_sac_pendulum_ground_truth(target, config=config, seed=seed)
            target_runs["SAC (ground truth)"].append(sac_run)
            all_runs.append(sac_run)

            for budget in config.preference_budgets:
                pebble_run = train_pebble_pendulum(target, budget, config=config, seed=seed)
                target_runs[f"PEBBLE (budget={budget})"].append(pebble_run)
                all_runs.append(pebble_run)

        aggregates: dict[str, dict[str, object]] = {}
        for label, runs in target_runs.items():
            aggregates[label] = _aggregate_method_results(runs)
        aggregated_by_target[target] = aggregates

        _plot_target_comparison(
            target,
            aggregates,
            root / "plots" / f"target_{target}_comparison.png",
        )

    _plot_normalized_summary(
        aggregated_by_target,
        root / "plots" / "step5_normalized_summary.png",
    )

    summary = {
        "config": asdict(config),
        "aggregates_by_target": aggregated_by_target,
        "runs": [asdict(run) for run in all_runs],
    }
    save_json(summary, root / "summary.json")

    for run in all_runs:
        method_dir = "sac" if run.method == "sac_ground_truth" else f"pebble_budget_{run.preference_budget}"
        run_path = root / "logs" / f"target_{run.target_angle_deg}" / method_dir
        save_json(asdict(run), run_path / f"seed_{run.seed}.json")

    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Step 5 Pendulum SAC vs PEBBLE experiments.")
    parser.add_argument("--total-steps", type=int, default=200_000)
    parser.add_argument("--start-steps", type=int, default=10_000)
    parser.add_argument("--eval-interval", type=int, default=10_000)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--num-seeds", type=int, default=15)
    parser.add_argument("--seed-start", type=int, default=None, help="Inclusive start seed for chunked runs")
    parser.add_argument("--seed-end", type=int, default=None, help="Inclusive end seed for chunked runs")
    parser.add_argument("--seeds", type=int, nargs="+", default=None, help="Explicit seed list, overrides ranges")
    parser.add_argument(
        "--angles",
        type=int,
        nargs="+",
        default=list(PENDULUM_TARGET_ANGLES_DEG),
        help="Subset of target angles to run, e.g. --angles 0 or --angles 0 -60 90",
    )
    parser.add_argument("--output-dir", type=str, default="src/pebble/results/step5_pendulum")
    parser.add_argument("--budgets", type=int, nargs="+", default=[500, 1000, 3000])
    parser.add_argument("--device", type=str, choices=["cpu", "cuda"], default=_default_device())
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    selected_angles = tuple(int(angle) for angle in args.angles)
    invalid_angles = [angle for angle in selected_angles if angle not in PENDULUM_TARGET_ANGLES_DEG]
    if invalid_angles:
        raise ValueError(
            f"Unsupported angles: {invalid_angles}. Supported angles: {list(PENDULUM_TARGET_ANGLES_DEG)}"
        )

    if args.seeds is not None:
        selected_seeds = tuple(int(seed) for seed in args.seeds)
    elif args.seed_start is not None or args.seed_end is not None:
        if args.seed_start is None or args.seed_end is None:
            raise ValueError("Both --seed-start and --seed-end must be provided together")
        if args.seed_start > args.seed_end:
            raise ValueError("--seed-start must be <= --seed-end")
        selected_seeds = tuple(range(int(args.seed_start), int(args.seed_end) + 1))
    else:
        selected_seeds = tuple(range(args.num_seeds))

    config = Step5Config(
        seeds=selected_seeds,
        target_angles_deg=selected_angles,
        preference_budgets=tuple(int(x) for x in args.budgets),
        total_steps=args.total_steps,
        start_steps=args.start_steps,
        eval_interval=args.eval_interval,
        eval_episodes=args.eval_episodes,
        device=args.device,
    )
    run_step5_experiments(config=config, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
