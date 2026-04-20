"""Learned reward model for preference-based learning."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
import torch.optim as optim

from src.pebble.preference_buffer import PreferencePair, TrajectorySegment
from src.sac.models import build_mlp


@dataclass(slots=True)
class RewardModelConfig:
	"""Configuration for the learned reward model."""

	observation_dim: int
	action_dim: int
	hidden_sizes: tuple[int, ...] = (256, 256)
	learning_rate: float = 1e-3
	device: str = "cpu"


class RewardModel(nn.Module):
	"""Neural network that predicts reward (preference probability) for trajectory segments.
	
	The model takes two trajectory segments and predicts the probability that the first
	segment is preferred over the second based on learned preferences.
	"""

	def __init__(self, config: RewardModelConfig) -> None:
		"""Initialize the reward model.
		
		Args:
			config: RewardModelConfig with architecture and hyperparameters.
		"""
		super().__init__()
		self.config = config
		self.device = torch.device(config.device)

		# Input to the network: concatenated observation and action from both segments
		# Each segment contributes: [observation (obs_dim) + action (action_dim)]
		# For two segments: 2 * (obs_dim + action_dim)
		input_dim = 2 * (config.observation_dim + config.action_dim)

		# Network outputs a single logit for preference probability
		self.network = build_mlp(
			input_dim=input_dim,
			hidden_sizes=config.hidden_sizes,
			output_dim=1,
			activation=nn.ReLU,
		)

		# Sigmoid to convert logit to probability
		self.sigmoid = nn.Sigmoid()

		self.to(self.device)

	def forward(self, segment1_obs: torch.Tensor, segment1_action: torch.Tensor,
		segment2_obs: torch.Tensor, segment2_action: torch.Tensor) -> torch.Tensor:
		"""Forward pass to get preference probability.
		
		Args:
			segment1_obs: Observations from segment 1, shape (batch_size, obs_dim)
			segment1_action: Actions from segment 1, shape (batch_size, action_dim)
			segment2_obs: Observations from segment 2, shape (batch_size, obs_dim)
			segment2_action: Actions from segment 2, shape (batch_size, action_dim)
			
		Returns:
			Preference probability for segment 1 over segment 2, shape (batch_size, 1)
		"""
		# Concatenate features from both segments
		combined = torch.cat([segment1_obs, segment1_action, segment2_obs, segment2_action], dim=1)
		
		# Pass through network
		logits = self.network(combined)
		
		# Convert to probability
		probs = self.sigmoid(logits)
		
		return probs

	def predict_preference(
		self, segment1_obs: np.ndarray, segment1_action: np.ndarray,
		segment2_obs: np.ndarray, segment2_action: np.ndarray,
	) -> np.ndarray:
		"""Predict preference probability for a batch without backprop.
		
		Args:
			segment1_obs: Batch of observations from segment 1, shape (batch_size, obs_dim)
			segment1_action: Batch of actions from segment 1, shape (batch_size, action_dim)
			segment2_obs: Batch of observations from segment 2, shape (batch_size, obs_dim)
			segment2_action: Batch of actions from segment 2, shape (batch_size, action_dim)
			
		Returns:
			Numpy array of preference probabilities, shape (batch_size,)
		"""
		with torch.no_grad():
			s1_obs = torch.tensor(segment1_obs, dtype=torch.float32, device=self.device)
			s1_act = torch.tensor(segment1_action, dtype=torch.float32, device=self.device)
			s2_obs = torch.tensor(segment2_obs, dtype=torch.float32, device=self.device)
			s2_act = torch.tensor(segment2_action, dtype=torch.float32, device=self.device)

			probs = self.forward(s1_obs, s1_act, s2_obs, s2_act)
			return probs.cpu().numpy().flatten()

	def predict_batch_pairs(self, pairs: list[PreferencePair]) -> np.ndarray:
		"""Predict preference probabilities for a list of preference pairs.
		
		Args:
			pairs: List of PreferencePair objects.
			
		Returns:
			Array of preference probabilities for segment 1 over segment 2.
		"""
		if not pairs:
			return np.array([], dtype=np.float32)

		# Extract features from all pairs
		seg1_obs = np.array([p.segment_1.observation for p in pairs], dtype=np.float32)
		seg1_act = np.array([p.segment_1.action for p in pairs], dtype=np.float32)
		seg2_obs = np.array([p.segment_2.observation for p in pairs], dtype=np.float32)
		seg2_act = np.array([p.segment_2.action for p in pairs], dtype=np.float32)

		return self.predict_preference(seg1_obs, seg1_act, seg2_obs, seg2_act)


class RewardModelTrainer:
	"""Trainer for the learned reward model using preference labels."""

	def __init__(self, model: RewardModel, learning_rate: float | None = None) -> None:
		"""Initialize the trainer.
		
		Args:
			model: The RewardModel to train.
			learning_rate: Learning rate for the optimizer. If None, uses model config.
		"""
		self.model = model
		self.device = model.device
		
		lr = learning_rate or model.config.learning_rate
		self.optimizer = optim.Adam(model.parameters(), lr=lr)
		self.criterion = nn.BCELoss()

	def train_step(
		self,
		segment1_obs: np.ndarray,
		segment1_action: np.ndarray,
		segment2_obs: np.ndarray,
		segment2_action: np.ndarray,
		preferences: np.ndarray,
	) -> float:
		"""Train the reward model on a batch of preference pairs.
		
		Args:
			segment1_obs: Batch of observations from segment 1.
			segment1_action: Batch of actions from segment 1.
			segment2_obs: Batch of observations from segment 2.
			segment2_action: Batch of actions from segment 2.
			preferences: Preference labels (1 if segment1 preferred, 0 otherwise).
			
		Returns:
			The loss value for this batch.
		"""
		# Convert to tensors
		s1_obs = torch.tensor(segment1_obs, dtype=torch.float32, device=self.device)
		s1_act = torch.tensor(segment1_action, dtype=torch.float32, device=self.device)
		s2_obs = torch.tensor(segment2_obs, dtype=torch.float32, device=self.device)
		s2_act = torch.tensor(segment2_action, dtype=torch.float32, device=self.device)
		prefs = torch.tensor(preferences, dtype=torch.float32, device=self.device).reshape(-1, 1)

		# Forward pass
		predictions = self.model(s1_obs, s1_act, s2_obs, s2_act)

		# Compute loss
		loss = self.criterion(predictions, prefs)

		# Backward pass
		self.optimizer.zero_grad()
		loss.backward()
		self.optimizer.step()

		return float(loss.item())

	def train_on_pairs(
		self,
		pairs: list[PreferencePair],
		preferences: np.ndarray,
		num_epochs: int = 10,
		batch_size: int = 32,
	) -> list[float]:
		"""Train the reward model on a list of preference pairs.
		
		Args:
			pairs: List of PreferencePair objects.
			preferences: Preference labels for each pair.
			num_epochs: Number of training epochs.
			batch_size: Batch size for training.
			
		Returns:
			List of average losses per epoch.
		"""
		losses_per_epoch = []
		n_pairs = len(pairs)

		for epoch in range(num_epochs):
			epoch_loss = 0.0
			num_batches = 0

			# Shuffle indices
			indices = np.random.permutation(n_pairs)

			# Mini-batch training
			for i in range(0, n_pairs, batch_size):
				batch_indices = indices[i : i + batch_size]

				# Extract batch features
				batch_pairs = [pairs[idx] for idx in batch_indices]
				batch_prefs = preferences[batch_indices]

				seg1_obs = np.array([p.segment_1.observation for p in batch_pairs], dtype=np.float32)
				seg1_act = np.array([p.segment_1.action for p in batch_pairs], dtype=np.float32)
				seg2_obs = np.array([p.segment_2.observation for p in batch_pairs], dtype=np.float32)
				seg2_act = np.array([p.segment_2.action for p in batch_pairs], dtype=np.float32)

				# Train step
				loss = self.train_step(seg1_obs, seg1_act, seg2_obs, seg2_act, batch_prefs)
				epoch_loss += loss
				num_batches += 1

			avg_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
			losses_per_epoch.append(avg_loss)

		return losses_per_epoch

	def save_model(self, path: str) -> None:
		"""Save the model state dict to a file.
		
		Args:
			path: Path to save the model.
		"""
		torch.save(self.model.state_dict(), path)

	def load_model(self, path: str) -> None:
		"""Load the model state dict from a file.
		
		Args:
			path: Path to load the model from.
		"""
		self.model.load_state_dict(torch.load(path, map_location=self.device))
