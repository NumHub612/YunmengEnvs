# -*- encoding: utf-8 -*-
"""
Copyright (C) 2026, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

description
"""

from __future__ import annotations

import datetime
from typing import Any, Mapping

import numpy as np
import torch

from yunmeng.interfaces.capabilities import IDifferentiable, IEstimable
from yunmeng.interfaces.types import ArrayLike
from yunmeng.interfaces.capabilities import (
    EstimationResult,
    IEstimator,
    ILoss,
    IObservationSet,
    MemoryStrategy,
    ModelRef,
    TrainingMeta,
)


class TrajectoryObservationSet:
    """Full-field trajectory observations, shaped (n_steps, n_cells)."""

    def __init__(self, times: ArrayLike, fields: dict[str, ArrayLike]):
        self._times = np.asarray(times, dtype="float64")
        self._fields = {
            name: np.asarray(values, dtype="float64") for name, values in fields.items()
        }

    def variables(self) -> list[str]:
        return list(self._fields)

    def times(self) -> ArrayLike:
        return self._times

    def values(self, variable: str) -> ArrayLike:
        return self._fields[variable]


class TrajectoryMSELoss:
    """MSE between a predicted trajectory and one observed field."""

    def __init__(self, variable: str):
        self._variable = variable

    def __call__(self, predicted: Any, observed: IObservationSet) -> ArrayLike:
        observed_values = observed.values(self._variable)
        if isinstance(predicted, torch.Tensor):
            target = torch.as_tensor(
                observed_values,
                dtype=predicted.dtype,
                device=predicted.device,
            )
            return torch.mean((predicted - target) ** 2)
        return np.mean((np.asarray(predicted) - observed_values) ** 2)


def _window(data: IObservationSet, start: int, end: int):
    return TrajectoryObservationSet(
        times=data.times()[start:end],
        fields={
            variable: data.values(variable)[start:end] for variable in data.variables()
        },
    )


class GradientTrainer(IEstimator):
    """Unrolled-AD trainer using the target's IDifferentiable leaves."""

    def __init__(
        self,
        lr: float = 1e-3,
        epochs: int = 100,
        n_steps: int = 50,
        memory: MemoryStrategy = MemoryStrategy.FULL,
        device: str = "cpu",
        val_fraction: float = 0.2,
        patience: int = 60,
    ):
        if memory != MemoryStrategy.FULL:
            raise NotImplementedError(f"{memory.value} is not implemented")
        if epochs < 1:
            raise ValueError("epochs must be positive")
        self._lr = lr
        self._epochs = epochs
        self._n_steps = n_steps
        self._device = device
        self._val_fraction = val_fraction
        self._patience = patience

    @classmethod
    def get_name(cls) -> str:
        return "GradientTrainer(full)"

    def _grad_leaves(self, target: IEstimable) -> list:
        if not isinstance(target, IDifferentiable):
            return []
        return list(target.grad_parameters())

    def _reset_target(self, target: IEstimable):
        if hasattr(target, "reset"):
            target.reset()
        elif hasattr(target, "reset_run"):
            target.reset_run()
        else:
            raise RuntimeError("estimation target must provide reset() or reset_run()")

    def fit(
        self,
        target: IEstimable,
        data: IObservationSet,
        loss: ILoss,
        **kwargs,
    ) -> EstimationResult:
        if not target.supports_gradients():
            raise RuntimeError(
                "target reports no end-to-end gradient support; use a "
                "gradient-free calibrator instead"
            )
        leaves = self._grad_leaves(target)
        if not leaves:
            raise RuntimeError("no differentiable parameters found on target")
        for parameter in leaves:
            parameter.requires_grad_(True)

        optimizer = torch.optim.Adam(leaves, lr=self._lr)
        n_steps = int(kwargs.get("n_steps", self._n_steps))
        n_val = max(1, int(n_steps * self._val_fraction))
        if n_steps > 1:
            n_val = min(n_val, n_steps - 1)
        n_fit = n_steps - n_val
        if n_fit <= 0:
            n_fit = n_steps
            n_val = 0

        history = []
        best_val = float("inf")
        best_state = None
        stall = 0

        if hasattr(target, "train"):
            target.train()
        try:
            for epoch in range(self._epochs):
                optimizer.zero_grad()
                self._reset_target(target)
                predicted = target.run(n_steps)

                train_loss = loss(predicted[:n_fit], _window(data, 0, n_fit))
                train_loss.backward()
                optimizer.step()

                with torch.no_grad():
                    if n_val:
                        val_prediction = predicted[n_fit:].detach()
                        val_data = _window(data, n_fit, n_steps)
                    else:
                        val_prediction = predicted.detach()
                        val_data = data
                    val_loss = float(loss(val_prediction, val_data))

                history.append(
                    {
                        "epoch": epoch,
                        "loss": float(train_loss.item()),
                        "val_loss": val_loss,
                    }
                )
                if val_loss < best_val - 1e-12:
                    best_val = val_loss
                    best_state = np.array(target.get_parameters(), copy=True)
                    stall = 0
                else:
                    stall += 1
                    if stall >= self._patience:
                        history.append({"epoch": epoch, "early_stop": True})
                        break
        finally:
            if hasattr(target, "eval"):
                target.eval()

        if best_state is not None:
            target.set_parameters(best_state)

        return EstimationResult(
            parameters={"names": target.parameter_names()},
            metrics={
                "final_loss": history[-1].get("loss", float("nan")),
                "best_val_loss": best_val,
            },
            history=history,
            converged=True,
            message=(
                f"{len(history)} epochs, full unrolled AD, "
                f"validation tail {n_val}/{n_steps} steps"
            ),
        )

    def evaluate(
        self,
        target: IEstimable,
        data: IObservationSet,
        **kwargs,
    ) -> dict[str, float]:
        self._reset_target(target)
        n_steps = int(kwargs.get("n_steps", len(data.times())))
        with torch.no_grad():
            predicted = target.run(n_steps)
        if isinstance(predicted, torch.Tensor):
            predicted = predicted.detach().cpu().numpy()
        else:
            predicted = np.asarray(predicted)
        observed = data.values(data.variables()[0])[: predicted.shape[0]]
        return {
            "rmse": float(np.sqrt(np.mean((predicted - observed) ** 2))),
            "mae": float(np.mean(np.abs(predicted - observed))),
            "n_steps": float(predicted.shape[0]),
        }


class InMemoryArtifactStore:
    """Versioned in-memory model-artifact store."""

    def __init__(self):
        self._store = {}

    def resolve(self, ref: ModelRef) -> dict[str, ArrayLike]:
        try:
            return self._store[ref.model_id][ref.version][0]
        except KeyError:
            raise KeyError(f"missing model artifact: {ref}") from None

    def register(
        self,
        params: Mapping[str, ArrayLike],
        meta: TrainingMeta,
        model_id: str = "model",
    ) -> ModelRef:
        versions = self._store.setdefault(model_id, {})
        version = f"v{len(versions) + 1}"
        versions[version] = (dict(params), meta)
        return ModelRef(model_id=model_id, version=version)

    def meta(self, ref: ModelRef) -> TrainingMeta:
        return self._store[ref.model_id][ref.version][1]

    def list_versions(self, model_id: str) -> list[str]:
        return list(self._store.get(model_id, {}))


def now_iso() -> str:
    return datetime.datetime.now().isoformat(timespec="seconds")
