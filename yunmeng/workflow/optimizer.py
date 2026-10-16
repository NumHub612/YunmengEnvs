# -*- encoding: utf-8 -*-
"""
Copyright (C) 2026, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Estimation implementations: observations, losses, trainers.
"""

from typing import Callable

import numpy as np

from yunmeng.standard.types import ArrayLike
from yunmeng.standard.capabilities import (
    IEstimator,
    IEstimable,
    IObservationSet,
    EstimationResult,
)
from yunmeng.numerics.algos import ym_register

try:
    import torch

    _HAS_TORCH = True
except ImportError:  # pragma: no cover
    torch = None
    _HAS_TORCH = False

# ---------------------------------------------------
# region Trajectory Kitts
# ---------------------------------------------------


class TrajectorySet:
    """Observation set: full-field trajectory (n_steps, n_cells)."""

    def __init__(self, times: ArrayLike, fields: dict):
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

    def __init__(self, variable: str):
        self._var = variable

    def __call__(self, predicted: ArrayLike, observed: IObservationSet):
        obs = observed.values(self._var)
        if isinstance(predicted, torch.Tensor):
            target = torch.as_tensor(
                obs, dtype=predicted.dtype, device=predicted.device
            )
            return torch.mean((predicted - target) ** 2)
        return np.mean((np.asarray(predicted) - obs) ** 2)


class TrajectoryRMSELoss:
    """RMSE metric for evaluation. For training prefer TrajectoryMSELoss:
    sqrt is a monotone transform of MSE (same argmin) but its gradient
    diverges as MSE -> 0, which destabilizes late-stage training."""

    def __init__(self, variable: str):
        self._mse = TrajectoryMSELoss(variable)

    def __call__(self, predicted: ArrayLike, observed: IObservationSet):
        mse = self._mse(predicted, observed)
        return torch.sqrt(mse) if isinstance(mse, torch.Tensor) else float(np.sqrt(mse))


def _window(data: TrajectorySet, start: int, end: int) -> TrajectorySet:
    return TrajectorySet(
        times=data.times()[start:end],
        fields={v: data.values(v)[start:end] for v in data.variables()},
    )


# ---------------------------------------------------
# region GradientTrainer
# ---------------------------------------------------


@ym_register("estimator")
class GradientTrainer(IEstimator):
    """Unrolled-AD trainer: full backpropagation through the rollout.

    Fits the first ``n_fit`` steps and early-stops on the tail
    ``n_val`` steps of the same rollout window. The best tail-window
    state is restored after training.
    """

    def __init__(
        self,
        lr: float = 1e-3,
        epochs: int = 100,
        n_steps: int = 50,
        device: str = "cpu",
        val_fraction: float = 0.2,
        patience: int = 60,
    ):
        """Initialize the trainer.

        Args:
            lr: learning rate
            epochs: number of training epochs
            n_steps: number of steps to train on
            device: torch device
            val_fraction: fraction of steps to use for validation
            patience: patience for early stopping
        """
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
        """Get the parameters of the target solver."""
        solver = getattr(target, "solver", target)
        gp = getattr(solver, "grad_parameters", None)
        return list(gp()) if callable(gp) else []

    def fit(
        self,
        target: IEstimable,
        data: TrajectorySet,
        loss: Callable,
    ) -> EstimationResult:
        if not target.supports_gradients():
            raise RuntimeError(
                "target reports no end-to-end gradient support; "
                "use a gradient-free Calibrator instead."
            )
        # Open grad mode for all parameters
        leaves = self._torch_params(target)
        if not leaves:
            raise RuntimeError("no differentiable parameters found on target")
        for p in leaves:
            p.requires_grad_(True)

        # Optimize
        opt = torch.optim.Adam(leaves, lr=self._lr)

        # Dataset split
        history = []
        n_val = max(1, int(self._n_steps * self._val_fraction))  # validation steps
        n_fit = self._n_steps - n_val  # fitting steps
        best_val = float("inf")
        best_state = None
        stall = 0

        # Training loop
        target.train()  # solver switch to training mode
        try:
            for epoch in range(self._epochs):
                # state reset
                opt.zero_grad()
                target.reset()
                # rollout timeseries result
                predicted = target.run(self._n_steps)
                # loss
                lval = loss(predicted[:n_fit], _window(data, 0, n_fit))
                lval.backward()
                opt.step()
                # validation in eval mode
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
                # early stopping
                if vval < best_val - 1e-12:
                    best_val = vval
                    best_state = np.array(target.get_parameters(), copy=True)
                    stall = 0
                else:
                    stall += 1
                    if stall >= self._patience:
                        history.append({"epoch": epoch, "early_stop": True})
                        break
        finally:
            # back to eval mode
            target.eval()

        # restore best state
        if best_state is not None:
            target.set_parameters(best_state)

        # return result
        return EstimationResult(
            parameters=dict(zip(target.parameter_names(), target.get_parameters())),
            metrics={
                "final_loss": history[-1].get("loss", float("nan")),
                "best_val_loss": best_val,
            },
            history=history,
            message=f"{len(history)} epochs, full unrolled AD, "
            f"early-stop on tail {n_val}/{self._n_steps} steps",
        )

    def evaluate(
        self,
        target: IEstimable,
        data: TrajectorySet,
        n_steps: int = None,
    ) -> dict:
        target.reset()
        n = n_steps or len(data.times())
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


ym_losses = {"mse": TrajectoryMSELoss, "rmse": TrajectoryRMSELoss}
