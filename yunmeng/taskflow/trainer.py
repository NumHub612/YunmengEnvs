# -*- encoding: utf-8 -*-
"""
GradientTrainer: unrolled-AD trainer (v1.1 §13.1 / v2.0 §7).

fit() contract:
- entry checks supports_gradients(), fails loudly with a Calibrator hint;
- sets target to TRAIN, restores EVAL on exit or exception;
- unroll solver steps via target.run(unroll_steps); loss.backward() into
  theta; Adam step. MemoryStrategy.FULL (short rollouts) is implemented;
  CHECKPOINT/ADJOINT keep the v1.1 enum semantics (reserved).

Parameter collection:
- neural operators: their nn.Module parameter tensors directly
  (optimizer must hold the SAME tensor objects used in forward);
- phys.* scalars listed in `train_phys`: wrapped as requires-grad leaf
  tensors and routed into the mechanism operators via set_parameters
  (mechanism ops store references, so identity is preserved).
"""

from __future__ import annotations

from typing import Any

import torch

from yunmeng.interfaces.solution import (
    EstimationResult,
    IEstimable,
    MemoryStrategy,
)
from yunmeng.interfaces.types import RunMode
from .dataset import TrajectoryDataset
from .losses import rollout_mse


class GradientTrainer:
    """End-to-end trainer over unrolled solver steps."""

    def __init__(
        self,
        unroll_steps: int = 20,
        epochs: int = 200,
        lr: float = 1e-3,
        memory_strategy: MemoryStrategy = MemoryStrategy.FULL,
        train_phys: list[str] | None = None,
    ):
        if memory_strategy is not MemoryStrategy.FULL:
            raise NotImplementedError(
                "This demo implements MemoryStrategy.FULL only "
                "(CHECKPOINT/ADJOINT reserved per v1.1 §13.1)."
            )
        self._unroll = int(unroll_steps)
        self._epochs = int(epochs)
        self._lr = float(lr)
        self._train_phys = list(train_phys or [])

    @classmethod
    def get_name(cls) -> str:
        return "GradientTrainer"

    # -- IEstimator ----------------------------------------

    def fit(
        self,
        target: IEstimable,
        data: TrajectoryDataset,
        loss=rollout_mse,
        **kwargs,
    ) -> EstimationResult:
        if not isinstance(target, IEstimable):
            raise TypeError(
                f"{type(target).__name__} is not IEstimable; "
                "mix the estimation contract into the model first."
            )
        if not target.supports_gradients():
            raise TypeError(
                f"{type(target).__name__} does not support gradients. "
                "Use Calibrator for black-box targets."
            )

        theta, nn_modules = self._collect_theta(target)
        if not theta:
            raise ValueError("No trainable parameters found.")
        optim = torch.optim.Adam(theta, lr=self._lr)

        result = EstimationResult()
        target.train()
        try:
            for epoch in range(self._epochs):
                epoch_loss = 0.0
                for ep in data.episodes:
                    target.set_initial_condition(ep.ic)
                    target.reset_run()
                    traj = target.run(self._unroll, dt=ep.dt)
                    ref = torch.as_tensor(ep.reference, dtype=torch.float64)
                    l = loss(traj, ref)
                    optim.zero_grad()
                    l.backward()
                    optim.step()
                    epoch_loss += float(l)
                epoch_loss /= len(data.episodes)
                result.history.append({"epoch": epoch, "loss": epoch_loss})
        finally:
            target.eval()

        result.parameters = target.get_parameters()
        result.metrics = {
            "final_loss": result.history[-1]["loss"],
            "initial_loss": result.history[0]["loss"],
        }
        result.converged = True
        result.message = (
            f"GradientTrainer: {self._epochs} epochs, "
            f"loss {result.metrics['initial_loss']:.3e} -> "
            f"{result.metrics['final_loss']:.3e}."
        )
        return result

    def evaluate(self, target: IEstimable, data: TrajectoryDataset, **kwargs) -> dict:
        losses = []
        with torch.no_grad():
            for ep in data.episodes:
                target.set_initial_condition(ep.ic)
                target.reset_run()
                traj = target.run(self._unroll, dt=ep.dt)
                ref = torch.as_tensor(ep.reference, dtype=torch.float64)
                losses.append(float(rollout_mse(traj, ref)))
        return {"rollout_mse": sum(losses) / len(losses)}

    # -- internals -------------------------------------------

    def _collect_theta(self, target: IEstimable):
        """Collect optimizer tensors: nn params + selected phys scalars."""
        theta: list[torch.Tensor] = []

        # neural operators: reach through the model to nn.Modules
        nn_modules = []
        solver = getattr(target, "solver", None)
        if solver is not None:
            for op in solver.operators:
                if isinstance(op, torch.nn.Module):
                    nn_modules.append(op)
                    theta.extend([p for p in op.parameters() if p.requires_grad])

        # phys scalars: create leaf tensors, route into mechanism ops
        if self._train_phys:
            current = target.get_parameters()
            wrap = {}
            for name in self._train_phys:
                v = current[name]
                val = float(v.detach()) if isinstance(v, torch.Tensor) else float(v)
                t = torch.tensor(val, dtype=torch.float64, requires_grad=True)
                wrap[name] = t
                theta.append(t)
            target.set_parameters(wrap)

        return theta, nn_modules
