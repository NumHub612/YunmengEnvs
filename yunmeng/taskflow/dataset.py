# -*- encoding: utf-8 -*-
"""
TrajectoryDataset: observation set for trajectory-matching training.

One episode = (initial condition, reference trajectory, dt).
The reference trajectory comes from a TEACHER run (fine-grid pure-physics
solver in this demo; could be real observations later) — same
IObservationSet role either way.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class TrajectoryEpisode:
    ic: np.ndarray  # (nx, ny) initial condition
    reference: np.ndarray  # (n_steps, nx, ny) reference trajectory
    dt: float


class TrajectoryDataset:
    def __init__(self, episodes: list[TrajectoryEpisode]):
        if not episodes:
            raise ValueError("TrajectoryDataset needs at least one episode.")
        self.episodes = episodes

    # -- persistence ----------------------------------

    def save(self, path: str) -> None:
        payload = {}
        for i, ep in enumerate(self.episodes):
            payload[f"ic_{i}"] = ep.ic
            payload[f"ref_{i}"] = ep.reference
            payload[f"dt_{i}"] = np.array(ep.dt)
        payload["n_episodes"] = np.array(len(self.episodes))
        np.savez_compressed(path, **payload)

    @classmethod
    def load(cls, path: str) -> "TrajectoryDataset":
        z = np.load(path)
        n = int(z["n_episodes"])
        eps = [
            TrajectoryEpisode(
                ic=z[f"ic_{i}"], reference=z[f"ref_{i}"], dt=float(z[f"dt_{i}"])
            )
            for i in range(n)
        ]
        return cls(eps)
