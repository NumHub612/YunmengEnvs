# -*- encoding: utf-8 -*-
"""
Copyright (C) 2026, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Estimation implementations: observations, losses, trainers, artifact store.
"""

import datetime
import numpy as np


from yunmeng.interfaces.types import ArrayLike
from yunmeng.interfaces.capabilities import (
    IEstimable,
    IEstimator,
    EstimationResult,
    ILoss,
    IObservationSet,
    MemoryStrategy,
    ModelRef,
    TrainingMeta,
)
from yunmeng.numerics.algos import ym_register

try:
    import torch

    _HAS_TORCH = True
except ImportError:  # pragma: no cover
    torch = None
    _HAS_TORCH = False


class TrajectoryObservationSet(IObservationSet):
    """Observation set: full-field trajectory (n_steps, n_cells)."""

    def __init__(self, times: ArrayLike, fields: dict) -> None:
        self._times = np.asarray(times, dtype="float64")
        self._fields = {k: np.asarray(v, dtype="float64") for k, v in fields.items()}

    def variables(self) -> list:
        return list(self._fields.keys())

    def times(self):
        return self._times

    def values(self, variable: str):
        return self._fields[variable]


class TrajectoryMSELoss:
    """MSE between a predicted trajectory and observed fields."""

    def __init__(self, variable: str) -> None:
        self._var = variable

    def __call__(self, predicted: ArrayLike, observed: TrajectoryObservationSet):
        obs = observed.values(self._var)
        if isinstance(predicted, torch.Tensor):
            target = torch.as_tensor(
                obs, dtype=predicted.dtype, device=predicted.device
            )
            return torch.mean((predicted - target) ** 2)
        return np.mean((np.asarray(predicted) - obs) ** 2)


class TrajectoryRMSELoss:
    """RMSE variant of the trajectory loss."""

    def __init__(self, variable: str) -> None:
        self._mse = TrajectoryMSELoss(variable)

    def __call__(self, predicted: ArrayLike, observed: TrajectoryObservationSet):
        mse = self._mse(predicted, observed)
        return torch.sqrt(mse) if isinstance(mse, torch.Tensor) else float(np.sqrt(mse))


def _window(
    data: TrajectoryObservationSet, start: int, end: int
) -> TrajectoryObservationSet:
    return TrajectoryObservationSet(
        times=data.times()[start:end],
        fields={v: data.values(v)[start:end] for v in data.variables()},
    )


@ym_register("estimator")
class GradientTrainer(IEstimator):
    """Unrolled-AD trainer (MemoryStrategy.FULL)."""

    def __init__(
        self,
        lr: float = 1e-3,
        epochs: int = 100,
        n_steps: int = 50,
        memory: MemoryStrategy | str = MemoryStrategy.FULL,
        device: str = "cpu",
        val_fraction: float = 0.2,
        patience: int = 60,
    ) -> None:
        if isinstance(memory, str):
            memory = MemoryStrategy(memory)
        if memory != MemoryStrategy.FULL:
            raise NotImplementedError(f"{memory.value} not implemented yet")
        self._lr = lr
        self._epochs = epochs
        self._n_steps = n_steps
        self._device = device
        self._val_fraction = val_fraction
        self._patience = patience

    @classmethod
    def get_name(cls) -> str:
        return "GradientTrainer"

    def _torch_params(self, target: IEstimable) -> list:
        solver = getattr(target, "solver", target)
        params = []
        for op in getattr(solver, "_ops", []):
            tp = getattr(op, "torch_parameters", None)
            if callable(tp):
                params.extend(tp())
        return params

    def fit(
        self, target: IEstimable, data: TrajectoryObservationSet, loss: ILoss, **kwargs
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

        history = []
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

        return EstimationResult(
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

    def evaluate(
        self, target: IEstimable, data: TrajectoryObservationSet, **kwargs
    ) -> dict:
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
        return {
            "rmse": float(np.sqrt(np.mean((pred - obs) ** 2))),
            "mae": float(np.mean(np.abs(pred - obs))),
            "n_steps": float(pred.shape[0]),
        }


class InMemoryArtifactStore:
    """Versioned in-memory store for parameters theta."""

    def __init__(self) -> None:
        self._store: dict = {}

    def resolve(self, ref: ModelRef) -> dict:
        try:
            return self._store[ref.model_id][ref.version][0]
        except KeyError:
            raise KeyError(f"missing model artifact: {ref}") from None

    def register(self, params: dict, meta: TrainingMeta) -> ModelRef:
        model_id = meta.notes or "model"
        versions = self._store.setdefault(model_id, {})
        version = f"v{len(versions) + 1}"
        versions[version] = (dict(params), meta)
        return ModelRef(model_id=model_id, version=version)

    def meta(self, ref: ModelRef) -> TrainingMeta:
        return self._store[ref.model_id][ref.version][1]

    def list_versions(self, model_id: str) -> list:
        return list(self._store.get(model_id, {}).keys())


def now_iso() -> str:
    return datetime.datetime.now().isoformat(timespec="seconds")


ym_losses = {"mse": TrajectoryMSELoss, "rmse": TrajectoryRMSELoss}
