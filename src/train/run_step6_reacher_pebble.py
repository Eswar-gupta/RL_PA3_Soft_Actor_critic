"""Step 6 experiments: Reacher PEBBLE with three simulated teachers (Ra, Rb, Rc)."""

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

from src.env_wrappers import make_reacher_env
from src.pebble import PebbleAgent, PebbleSACConfig, PreferenceBuffer, PreferenceTeacher
from src.replay_buffer import ReplayBuffer
from src.utils import ensure_dir, save_json, set_seed


def _default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


@dataclass(slots=True)
class Step6Config:
    seeds: tuple[int, ...] = tuple(range(15))
    teacher_formulations: tuple[str, ...] = ("Ra", "Rb", "Rc")
    evaluation_formulations: tuple[str, ...] = ("Ra", "Rb", "Rc")
    preference_budgets: tuple[int, ...] = (3000,)
    total_steps: int = 1_000_000
    start_steps: int = 10_000
    eval_interval: int = 10_000
    eval_episodes: int = 20
    max_episode_steps: int = 1000
    target_radius: float = 0.05
    velocity_threshold: float = 1e-2
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
    query_interval: int = 5_000
    query_batch_size: int = 20
    reward_model_epochs: int = 5
    reward_model_batch_size: int = 64


@dataclass(slots=True)
class Step6RunResult:
    teacher_formulation: str
    seed: int
    preference_budget: int
    preference_queries_used: int
    evaluation_steps: list[int]
    evaluation_curves: dict[str, list[float]]


class PairwiseLinearRewardModel(nn.Module):
    """Reward model trained via Bradley-Terry pairwise comparisons."""

    def __init__(self, observation_dim: int, action_dim: int, device: str) -> None:
        super().__init__()
        self.device = torch.device(device)
        self.linear = nn.Linear(observation_dim + action_dim, 1, bias=True)
        self.to(self.device)

    def _reward_tensor(self, observations: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        features = torch.cat([observations, actions], dim=1)
        return self.linear(features)

    def predict_reward(self, observations: np.ndarray, actions: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            obs = torch.as_tensor(observations, dtype=torch.float32, device=self.device)
            act = torch.as_tensor(actions, dtype=torch.float32, device=self.device)
            reward = self._reward_tensor(obs, act)
            return reward.cpu().numpy().reshape(-1, 1)

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
        f1_all = torch.as_tensor(pair_features_1, dtype=torch.float32, device=self.device)
        f2_all = torch.as_tensor(pair_features_2, dtype=torch.float32, device=self.device)
        y_all = torch.as_tensor(labels, dtype=torch.float32, device=self.device).reshape(-1, 1)

        n_samples = int(f1_all.shape[0])
        if n_samples == 0:
            return []

        losses: list[float] = []
        for _ in range(epochs):
            perm = torch.randperm(n_samples, device=self.device)
            epoch_loss = 0.0
            n_batches = 0
            for start in range(0, n_samples, batch_size):
                idx = perm[start : start + batch_size]
                f1 = f1_all[idx]
                f2 = f2_all[idx]
                y = y_all[idx]

                logits = self.linear(f1) - self.linear(f2)
                loss = F.binary_cross_entropy_with_logits(logits, y)

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

                epoch_loss += float(loss.item())
                n_batches += 1

            losses.append(epoch_loss / max(n_batches, 1))

        return losses


def _pair_features_from_preferences(pairs) -> tuple[np.ndarray, np.ndarray]:
    features_1: list[np.ndarray] = []
    features_2: list[np.ndarray] = []
    for pair in pairs:
        s1 = np.concatenate([pair.segment_1.observation, pair.segment_1.action], axis=0)
        s2 = np.concatenate([pair.segment_2.observation, pair.segment_2.action], axis=0)
        features_1.append(s1)
        features_2.append(s2)
    return np.asarray(features_1, dtype=np.float32), np.asarray(features_2, dtype=np.float32)


def evaluate_policy_on_reacher(
    agent: PebbleAgent,
    reward_formulation: str,
    *,
    episodes: int,
    max_episode_steps: int,
    target_radius: float,
    velocity_threshold: float,
    seed: int,
) -> float:
    env = make_reacher_env(
        reward_formulation,
        max_episode_steps=max_episode_steps,
        target_radius=target_radius,
        velocity_threshold=velocity_threshold,
        seed=seed,
    )
    returns: list[float] = []

    for episode_idx in range(episodes):
        observation, _ = env.reset(seed=seed + episode_idx)
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


def train_pebble_reacher_teacher(
    teacher_formulation: str,
    preference_budget: int,
    *,
    config: Step6Config,
    seed: int,
) -> Step6RunResult:
    set_seed(seed)

    env = make_reacher_env(
        teacher_formulation,
        max_episode_steps=config.max_episode_steps,
        target_radius=config.target_radius,
        velocity_threshold=config.velocity_threshold,
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
    teacher = PreferenceTeacher(lambda _obs, _act: 0.0)
    reward_model = PairwiseLinearRewardModel(observation_dim, action_dim, device=config.device)

    preference_queries_used = 0
    observation, _ = env.reset(seed=seed)

    evaluation_steps: list[int] = []
    evaluation_curves: dict[str, list[float]] = {
        key: [] for key in config.evaluation_formulations
    }

    for step in range(1, config.total_steps + 1):
        if step <= config.start_steps:
            action = action_space.sample()
        else:
            action = agent.select_action(observation, deterministic=False)

        next_observation, teacher_reward, terminated, truncated, _ = env.step(action)
        done = bool(terminated or truncated)

        replay_buffer.add(observation, action, teacher_reward, next_observation, done)
        preference_buffer.add_step(observation, action, teacher_reward, next_observation, done)
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
            for offset, eval_form in enumerate(config.evaluation_formulations):
                mean_ret = evaluate_policy_on_reacher(
                    agent,
                    eval_form,
                    episodes=config.eval_episodes,
                    max_episode_steps=config.max_episode_steps,
                    target_radius=config.target_radius,
                    velocity_threshold=config.velocity_threshold,
                    seed=seed + step + 10_000 * offset,
                )
                evaluation_curves[eval_form].append(mean_ret)

    env.close()
    return Step6RunResult(
        teacher_formulation=teacher_formulation,
        seed=seed,
        preference_budget=preference_budget,
        preference_queries_used=preference_queries_used,
        evaluation_steps=evaluation_steps,
        evaluation_curves=evaluation_curves,
    )


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


def _aggregate_runs(runs: list[Step6RunResult], eval_formulation: str) -> dict[str, object]:
    if not runs:
        return {}

    steps = runs[0].evaluation_steps
    curves = np.asarray([run.evaluation_curves[eval_formulation] for run in runs], dtype=np.float64)
    mean, low, high = _mean_ci(curves)

    auc_values = [
        float(np.trapz(np.asarray(run.evaluation_curves[eval_formulation], dtype=np.float64), x=np.asarray(steps, dtype=np.float64)))
        for run in runs
    ]
    final_values = [
        float(np.mean(run.evaluation_curves[eval_formulation][-10:])) if run.evaluation_curves[eval_formulation] else 0.0
        for run in runs
    ]

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
        "num_runs": len(runs),
    }


def _plot_eval_formulation_comparison(
    eval_formulation: str,
    aggregates: dict[str, dict[str, object]],
    output_path: Path,
) -> None:
    plt.figure(figsize=(10, 6))

    for label, data in aggregates.items():
        steps = np.asarray(data["evaluation_steps"], dtype=np.float64)
        mean = np.asarray(data["mean_curve"], dtype=np.float64)
        low = np.asarray(data["ci_low"], dtype=np.float64)
        high = np.asarray(data["ci_high"], dtype=np.float64)
        plt.plot(steps, mean, label=label)
        plt.fill_between(steps, low, high, alpha=0.2)

    plt.title(f"Step 6 Reacher: evaluation on {eval_formulation}")
    plt.xlabel("Environment timesteps")
    plt.ylabel("Average undiscounted return")
    plt.legend()
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def run_step6_experiments(
    *,
    config: Step6Config | None = None,
    output_dir: str | Path = "src/pebble/results/step6_reacher",
) -> dict[str, object]:
    config = config or Step6Config()
    root = ensure_dir(output_dir)

    all_runs: list[Step6RunResult] = []

    for teacher_form in config.teacher_formulations:
        for budget in config.preference_budgets:
            for seed in config.seeds:
                run = train_pebble_reacher_teacher(
                    teacher_form,
                    budget,
                    config=config,
                    seed=seed,
                )
                all_runs.append(run)

                run_dir = root / "logs" / f"teacher_{teacher_form.lower()}" / f"budget_{budget}"
                save_json(asdict(run), run_dir / f"seed_{seed}.json")

    aggregates_by_eval: dict[str, dict[str, dict[str, object]]] = {}
    for eval_form in config.evaluation_formulations:
        aggregates_by_eval[eval_form] = {}
        for teacher_form in config.teacher_formulations:
            for budget in config.preference_budgets:
                key = f"Teacher {teacher_form} (budget={budget})"
                selected = [
                    run
                    for run in all_runs
                    if run.teacher_formulation == teacher_form and run.preference_budget == budget
                ]
                aggregates_by_eval[eval_form][key] = _aggregate_runs(selected, eval_form)

        _plot_eval_formulation_comparison(
            eval_form,
            aggregates_by_eval[eval_form],
            root / "plots" / f"eval_{eval_form.lower()}_comparison.png",
        )

    summary = {
        "config": asdict(config),
        "aggregates_by_evaluation": aggregates_by_eval,
        "runs": [asdict(run) for run in all_runs],
    }
    save_json(summary, root / "summary.json")
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Step 6 Reacher PEBBLE teacher-comparison experiments.")
    parser.add_argument("--total-steps", type=int, default=1_000_000)
    parser.add_argument("--start-steps", type=int, default=10_000)
    parser.add_argument("--eval-interval", type=int, default=10_000)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--num-seeds", type=int, default=15)
    parser.add_argument("--budgets", type=int, nargs="+", default=[3000])
    parser.add_argument("--device", type=str, choices=["cpu", "cuda"], default=_default_device())
    parser.add_argument("--output-dir", type=str, default="src/pebble/results/step6_reacher")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    config = Step6Config(
        seeds=tuple(range(args.num_seeds)),
        preference_budgets=tuple(int(x) for x in args.budgets),
        total_steps=args.total_steps,
        start_steps=args.start_steps,
        eval_interval=args.eval_interval,
        eval_episodes=args.eval_episodes,
        device=args.device,
    )
    run_step6_experiments(config=config, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
