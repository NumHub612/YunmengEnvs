from __future__ import annotations

from .trainer import GradientTrainer
from .losses import rollout_mse
from .artifacts import LocalArtifactStore
from .dataset import TrajectoryEpisode, TrajectoryDataset

ym_estimators = {}
ym_losses = {}


def _register_unique(registry: dict, name: str, obj) -> None:
    if name in registry:
        raise ValueError(f"Duplicate registration: '{name}'.")
    registry[name] = obj


_register_unique(ym_estimators, "gradienttrainer", GradientTrainer)
_register_unique(ym_losses, "rollout_mse", rollout_mse)

__all__ = [
    "GradientTrainer",
    "rollout_mse",
    "LocalArtifactStore",
    "TrajectoryEpisode",
    "TrajectoryDataset",
    "ym_estimators",
    "ym_losses",
]
