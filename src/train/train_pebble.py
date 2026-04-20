"""Training loop for PEBBLE (learning from preferences) on Reacher."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np

from src.env_wrappers import make_reacher_env
from src.pebble import (
	PebbleAgent,
	PebbleSACConfig,
	PreferenceBuffer,
	PreferenceTeacher,
	RewardModel,
	RewardModelConfig,
	RewardModelTrainer,
)
from src.replay_buffer import ReplayBuffer
from src.sac.sac_agent import SACAgent
from src.utils import ensure_dir, save_json, set_seed


@dataclass(slots=True)
class PebbleEvaluationRecord:
	"""Record of a single evaluation episode."""

	step: int
	learned_reward: bool
	reward_formulation: str
	mean_return: float
	episode_returns: list[float]


def evaluate_policy_with_learned_rewards(
	agent: PebbleAgent,
	reward_model: RewardModel,
	preference_buffer: PreferenceBuffer,
	reward_formulation: str,
	*,
	episodes: int = 20,
	max_episode_steps: int = 1000,
	target_radius: float = 0.05,
	velocity_threshold: float = 1e-2,
	seed: int = 0,
) -> tuple[float, list[float]]:
	"""Evaluate policy using learned rewards from preference buffer."""

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
			# Use ground truth reward for evaluation
			return_value += float(reward)

		episode_returns.append(return_value)

	env.close()
	return float(np.mean(episode_returns)), episode_returns


def evaluate_policy_ground_truth(
	agent: PebbleAgent,
	reward_formulation: str,
	*,
	episodes: int = 20,
	max_episode_steps: int = 1000,
	target_radius: float = 0.05,
	velocity_threshold: float = 1e-2,
	seed: int = 0,
) -> tuple[float, list[float]]:
	"""Evaluate policy using ground-truth rewards."""

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


def train_pebble(
	teacher_reward_fn: Callable[[np.ndarray, np.ndarray], float],
	training_reward_formulation: str,
	*,
	seed: int = 0,
	total_steps: int = 1_000_000,
	start_steps: int = 10_000,
	batch_size: int = 256,
	buffer_capacity: int = 1_000_000,
	eval_interval: int = 10_000,
	eval_episodes: int = 20,
	preference_query_interval: int = 5_000,
	preference_batch_size: int = 10,
	reward_model_epochs: int = 10,
	max_episode_steps: int = 1000,
	target_radius: float = 0.05,
	velocity_threshold: float = 1e-2,
	gamma: float = 0.99,
	tau: float = 0.005,
	actor_lr: float = 3e-4,
	critic_lr: float = 3e-4,
	alpha_lr: float = 3e-4,
	reward_model_lr: float = 1e-3,
	hidden_sizes: tuple[int, ...] = (256, 256),
	target_entropy: float | None = None,
	device: str = "cpu",
	output_dir: str | Path | None = None,
	evaluation_formulations: Iterable[str] = ("Ra",),
) -> dict[str, object]:
	"""Train PEBBLE on Reacher with preference-based reward learning.
	
	Args:
		teacher_reward_fn: Ground-truth reward function for preference labels.
		training_reward_formulation: Which formulation to use for data collection.
		seed: Random seed.
		total_steps: Total environment steps.
		start_steps: Steps before policy training begins.
		batch_size: Replay buffer batch size for SAC updates.
		buffer_capacity: Replay buffer capacity.
		eval_interval: Evaluation frequency in steps.
		eval_episodes: Number of episodes per evaluation.
		preference_query_interval: How often to query preferences (in steps).
		preference_batch_size: Number of preference pairs to query at a time.
		reward_model_epochs: Training epochs for reward model per update.
		... (other SAC hyperparams)
		output_dir: Directory for saving results.
		evaluation_formulations: Reward formulations to evaluate on.
		
	Returns:
		Dictionary with training results.
	"""

	set_seed(seed)
	
	# Environment for data collection
	env = make_reacher_env(
		training_reward_formulation,
		max_episode_steps=max_episode_steps,
		target_radius=target_radius,
		velocity_threshold=velocity_threshold,
		seed=seed,
	)

	observation_dim = int(env.observation_space.shape[0])
	action_space = env.action_space
	action_dim = int(np.prod(action_space.shape))

	# PEBBLE components
	pebble_agent = PebbleAgent(
		PebbleSACConfig(
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

	# Preference learning components
	preference_buffer = PreferenceBuffer(capacity=buffer_capacity)
	teacher = PreferenceTeacher(teacher_reward_fn)
	reward_model_config = RewardModelConfig(
		observation_dim=observation_dim,
		action_dim=action_dim,
		hidden_sizes=hidden_sizes,
		learning_rate=reward_model_lr,
		device=device,
	)
	reward_model = RewardModel(reward_model_config)
	reward_model_trainer = RewardModelTrainer(reward_model)

	# Standard replay buffer for RL
	replay_buffer = ReplayBuffer(buffer_capacity, observation_dim, action_dim)

	observation, _ = env.reset(seed=seed)
	evaluation_steps: list[int] = []
	evaluation_records: list[PebbleEvaluationRecord] = []
	preference_query_steps: list[int] = []
	preference_pairs_queried: list[int] = []
	reward_model_losses: list[float] = []

	# Storage for learned rewards
	all_learned_rewards: dict[int, np.ndarray] = {}
	segment_to_buffer_index: dict[int, list[int]] = {}  # segment id -> replay buffer indices

	for step in range(1, total_steps + 1):
		# Environment interaction
		if step <= start_steps:
			action = action_space.sample()
		else:
			action = pebble_agent.select_action(observation, deterministic=False)

		next_observation, ground_truth_reward, terminated, truncated, _ = env.step(action)
		done = bool(terminated or truncated)

		# Add to preference buffer for preference learning
		preference_buffer.add_step(
			observation, action, ground_truth_reward, next_observation, done
		)

		# Add to replay buffer (initially with ground truth reward)
		replay_buffer.add(observation, action, ground_truth_reward, next_observation, done)

		observation = next_observation

		# Query preferences and train reward model
		if step % preference_query_interval == 0 and len(preference_buffer.segments) >= preference_batch_size * 2:
			# Sample preference pairs
			pairs = preference_buffer.sample_batch_preference_pairs(
				batch_size=preference_batch_size, gamma=gamma
			)
			
			if pairs:
				# Get preference labels from teacher
				preferences = teacher.label_batch(pairs)
				preference_query_steps.append(step)
				preference_pairs_queried.append(len(pairs))

				# Train reward model on preferences
				losses = reward_model_trainer.train_on_pairs(
					pairs, preferences, num_epochs=reward_model_epochs, batch_size=8
				)
				if losses:
					reward_model_losses.append(losses[-1])

		# SAC policy updates on replay buffer
		if step > start_steps and len(replay_buffer) >= batch_size:
			batch = replay_buffer.sample(batch_size)
			pebble_agent.update_from_batch(batch)

		if done:
			observation, _ = env.reset()

		# Periodic evaluation
		if step % eval_interval == 0:
			evaluation_steps.append(step)

			# Evaluate with ground-truth rewards
			for offset, formulation in enumerate(evaluation_formulations):
				mean_return, episode_returns = evaluate_policy_ground_truth(
					pebble_agent,
					formulation,
					episodes=eval_episodes,
					max_episode_steps=max_episode_steps,
					target_radius=target_radius,
					velocity_threshold=velocity_threshold,
					seed=seed + 1000 * offset + step,
				)
				evaluation_records.append(
					PebbleEvaluationRecord(
						step=step,
						learned_reward=False,
						reward_formulation=formulation,
						mean_return=mean_return,
						episode_returns=episode_returns,
					)
				)

	env.close()

	result = {
		"training_formulation": training_reward_formulation,
		"seed": seed,
		"total_steps": total_steps,
		"evaluation_steps": evaluation_steps,
		"preference_query_steps": preference_query_steps,
		"preference_pairs_queried": preference_pairs_queried,
		"reward_model_losses": reward_model_losses,
		"evaluation_records": [asdict(record) for record in evaluation_records],
	}

	if output_dir is not None:
		output_path = ensure_dir(output_dir)
		save_json(
			result,
			output_path / f"pebble_{training_reward_formulation.lower()}_seed_{seed}.json",
		)

	return result


def build_arg_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="Train PEBBLE on Reacher.")
	parser.add_argument(
		"--training-formulation",
		default="Ra",
		choices=["Ra", "Rb", "Rc", "ra", "rb", "rc"],
		help="Reward formulation for teacher preference labeling",
	)
	parser.add_argument("--seed", type=int, default=0)
	parser.add_argument("--total-steps", type=int, default=1_000_000)
	parser.add_argument("--start-steps", type=int, default=10_000)
	parser.add_argument("--eval-interval", type=int, default=10_000)
	parser.add_argument("--eval-episodes", type=int, default=20)
	parser.add_argument("--preference-query-interval", type=int, default=5_000)
	parser.add_argument("--preference-batch-size", type=int, default=10)
	parser.add_argument("--reward-model-epochs", type=int, default=10)
	parser.add_argument("--max-episode-steps", type=int, default=1000)
	parser.add_argument("--output-dir", type=str, default="experiments/pebble")
	return parser


def main() -> None:
	parser = build_arg_parser()
	args = parser.parse_args()

	# For now, using Ra reward function as example teacher
	from src.rewards.reacher_rewards import get_reward_function

	teacher_fn = get_reward_function("Ra")

	train_pebble(
		teacher_fn,
		args.training_formulation,
		seed=args.seed,
		total_steps=args.total_steps,
		start_steps=args.start_steps,
		eval_interval=args.eval_interval,
		eval_episodes=args.eval_episodes,
		preference_query_interval=args.preference_query_interval,
		preference_batch_size=args.preference_batch_size,
		reward_model_epochs=args.reward_model_epochs,
		max_episode_steps=args.max_episode_steps,
		output_dir=args.output_dir,
		evaluation_formulations=["Ra", "Rb", "Rc"],
	)


if __name__ == "__main__":
	main()
