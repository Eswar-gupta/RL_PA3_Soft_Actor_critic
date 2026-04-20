"""Shared utility helpers."""

from __future__ import annotations

import json
import random
from pathlib import Path
from statistics import NormalDist
from typing import Any

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and PyTorch."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def soft_update(target: torch.nn.Module, source: torch.nn.Module, tau: float) -> None:
    """Polyak average ``source`` into ``target``."""

    if not 0.0 <= tau <= 1.0:
        raise ValueError("tau must lie in [0, 1]")

    with torch.no_grad():
        for target_param, source_param in zip(target.parameters(), source.parameters(), strict=True):
            target_param.data.mul_(1.0 - tau)
            target_param.data.add_(source_param.data, alpha=tau)


def hard_update(target: torch.nn.Module, source: torch.nn.Module) -> None:
    """Copy ``source`` parameters into ``target``."""

    target.load_state_dict(source.state_dict())


def to_tensor(array: Any, device: torch.device) -> torch.Tensor:
    """Convert a numpy array or scalar to a float tensor on ``device``."""

    return torch.as_tensor(array, dtype=torch.float32, device=device)


def ensure_dir(path: str | Path) -> Path:
    """Create ``path`` if needed and return it as a ``Path``."""

    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def save_json(data: Any, path: str | Path) -> None:
    """Write JSON with a stable, human-readable layout."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=True)


def mean_and_confidence_interval(values: list[float], confidence: float = 0.95) -> tuple[float, float, float]:
    """Return mean and a symmetric normal-approximation confidence interval."""

    if not values:
        return 0.0, 0.0, 0.0

    array = np.asarray(values, dtype=np.float64)
    mean = float(array.mean())
    if array.size == 1:
        return mean, mean, mean

    std = float(array.std(ddof=1))
    sem = std / float(np.sqrt(array.size))
    z_value = float(NormalDist().inv_cdf((1.0 + confidence) / 2.0))
    interval = z_value * sem
    return mean, mean - interval, mean + interval

