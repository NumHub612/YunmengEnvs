# -*- encoding: utf-8 -*-
"""
Copyright (C) 2026, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Estimation layer for the hybrid-validation demo:
  - TrajectoryObservationSet (IObservationSet)
  - TrajectoryMSELoss        (ILoss)
  - GradientTrainer          (IEstimator; unrolled AD)
  - InMemoryArtifactStore    (IModelArtifactStore)
"""

from __future__ import annotations

import datetime
from typing import Any, Mapping

import numpy as np

from yunmeng.solvers.hyb.estimation import (
    EstimationResult,
    IEstimable,
    IEstimator,
    ILoss,
    IObservationSet,
    MemoryStrategy,
    ModelRef,
    TrainingMeta,
)
from yunmeng.interfaces.types import ArrayLike, RunMode

import torch

# ---------------------------------------------------
# region Observations & loss
# ---------------------------------------------------


class TrajectoryObservationSet:
    """Observation set: full-field trajectory (n_steps, n_cells)."""

    def __init__(self, times: ArrayLike, fields: dict[str, ArrayLike]):
        self._times = np.asarray(times, dtype="float64")
        self._fields = {k: np.asarray(v, dtype="float64") for k, v in fields.items()}

    def variables(self) -> list[str]:
        return list(self._fields.keys())

    def times(self) -> ArrayLike:
        return self._times

    def values(self, variable: str) -> ArrayLike:
        return self._fields[variable]


class TrajectoryMSELoss:
    """MSE between a predicted trajectory and observed fields."""

    def __init__(self, variable: str):
        self._var = variable

    def __call__(self, predicted: Any, observed: IObservationSet) -> ArrayLike:
        obs = observed.values(self._var)
        if isinstance(predicted, torch.Tensor):
            target = torch.as_tensor(
                obs, dtype=predicted.dtype, device=predicted.device
            )
            return torch.mean((predicted - target) ** 2)
        return np.mean((np.asarray(predicted) - obs) ** 2)


def _window(data: IObservationSet, start: int, end: int) -> "TrajectoryObservationSet":
    """Slice an observation set to a time window."""
    return TrajectoryObservationSet(
        times=data.times()[start:end],
        fields={v: data.values(v)[start:end] for v in data.variables()},
    )


# ---------------------------------------------------
# region GradientTrainer
# ---------------------------------------------------


class GradientTrainer(IEstimator):
    """Unrolled-AD trainer (MemoryStrategy.FULL for this small demo).

    fit() sets the target to TRAIN (mode propagation is the target's job)
    and restores EVAL on exit or exception. Entry checks
    supports_gradients() and suggests an alternative when unsatisfied.
    """

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
            raise NotImplementedError(f"{memory.value} not implemented in this demo")
        self._lr = lr
        self._epochs = epochs
        self._n_steps = n_steps
        self._device = device
        self._val_fraction = val_fraction
        self._patience = patience

    @classmethod
    def get_name(cls) -> str:
        return "GradientTrainer(full)"

    def _torch_params(self, target: IEstimable) -> list["torch.nn.Parameter"]:
        """Reach the differentiable leaves of the target's operators."""
        ops = getattr(target, "_ops", [])
        params: list[torch.nn.Parameter] = []
        for op in ops:
            tp = getattr(op, "torch_parameters", None)
            if callable(tp):
                params.extend(tp())
        return params

    def fit(
        self,
        target: IEstimable,
        data: IObservationSet,
        loss: ILoss,
        **kwargs,
    ) -> EstimationResult:
        if not target.supports_gradients():
            raise RuntimeError(
                "target reports no end-to-end gradient support; "
                "use a gradient-free Calibrator (e.g. SCE-UA) instead."
            )
        leaves = self._torch_params(target)
        if not leaves:
            raise RuntimeError("no differentiable parameters found on target")
        for p in leaves:
            p.requires_grad_(True)
        opt = torch.optim.Adam(leaves, lr=self._lr)

        history: list[dict] = []
        n_val = max(1, int(self._n_steps * self._val_fraction))
        n_fit = self._n_steps - n_val
        best_val = float("inf")
        best_state = None
        stall = 0

        target.train()
        try:
            for epoch in range(self._epochs):
                opt.zero_grad()
                target.reset_run()
                predicted = target.run(self._n_steps)
                # Fit on the early window; validate on the tail of the
                # SAME rollout (no extra forward pass).
                lval = loss(predicted[:n_fit], _window(data, 0, n_fit))
                lval.backward()
                opt.step()
                with torch.no_grad():
                    vval = float(
                        loss(
                            predicted[n_fit:].detach(),
                            _window(data, n_fit, self._n_steps),
                        )
                    )
                history.append(
                    {"epoch": epoch, "loss": float(lval.item()), "val_loss": vval}
                )
                if vval < best_val - 1e-12:
                    best_val = vval
                    best_state = np.array(target.get_param_vector(), copy=True)
                    stall = 0
                else:
                    stall += 1
                    if stall >= self._patience:
                        history.append({"epoch": epoch, "early_stop": True})
                        break
        finally:
            target.eval()

        if best_state is not None:
            target.set_param_vector(best_state)

        result = EstimationResult(
            parameters={"names": target.param_names()},
            metrics={
                "final_loss": history[-1].get("loss", float("nan")),
                "best_val_loss": best_val,
            },
            history=history,
            converged=True,
            message=f"{len(history)} epochs, full unrolled AD, "
            f"early-stop on tail {n_val}/{self._n_steps} steps",
        )
        return result

    def evaluate(
        self,
        target: IEstimable,
        data: IObservationSet,
        **kwargs,
    ) -> dict[str, float]:
        target.reset_run()
        n = kwargs.get("n_steps", len(data.times()))
        with torch.no_grad():
            predicted = target.run(n)
        pred = (
            predicted.detach().cpu().numpy()
            if isinstance(predicted, torch.Tensor)
            else np.asarray(predicted)
        )
        obs = data.values(data.variables()[0])[: pred.shape[0]]
        rmse = float(np.sqrt(np.mean((pred - obs) ** 2)))
        mae = float(np.mean(np.abs(pred - obs)))
        return {"rmse": rmse, "mae": mae, "n_steps": float(pred.shape[0])}


# ---------------------------------------------------
# region Artifact store
# ---------------------------------------------------


class InMemoryArtifactStore:
    """Versioned in-memory store for parameters θ (IModelArtifactStore)."""

    def __init__(self):
        self._store: dict[str, dict[str, tuple[dict[str, ArrayLike], TrainingMeta]]] = (
            {}
        )

    def resolve(self, ref: ModelRef) -> dict[str, ArrayLike]:
        try:
            return self._store[ref.model_id][ref.version][0]
        except KeyError:
            raise KeyError(f"missing model artifact: {ref}") from None

    def register(self, params: Mapping[str, ArrayLike], meta: TrainingMeta) -> ModelRef:
        model_id = meta.notes or "model"
        versions = self._store.setdefault(model_id, {})
        version = f"v{len(versions) + 1}"
        versions[version] = (dict(params), meta)
        return ModelRef(model_id=model_id, version=version)

    def meta(self, ref: ModelRef) -> TrainingMeta:
        return self._store[ref.model_id][ref.version][1]

    def list_versions(self, model_id: str) -> list[str]:
        return list(self._store.get(model_id, {}).keys())


def now_iso() -> str:
    return datetime.datetime.now().isoformat(timespec="seconds")
