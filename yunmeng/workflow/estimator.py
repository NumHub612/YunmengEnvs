# -*- encoding: utf-8 -*-
"""
Copyright (C) 2026, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Estimation layer.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable
import numpy as np

from yunmeng.solutions.standards import IEstimable
from yunmeng.setting import logger

# ---------------------------------------------------
# region EstimationResult
# ---------------------------------------------------


@dataclass
class EstimationResult:
    """Outcome of one estimation (calibration or training) run."""

    parameters: dict[str, Any] = field(default_factory=dict)
    """Inferred parameters theta (model refs, coefficients, ...)."""

    model_refs: list[tuple[str, str]] = field(default_factory=list)
    """Updated model artifact references: (model_id, version)."""

    metrics: dict[str, float] = field(default_factory=dict)
    """Evaluation metrics on validation/observation data."""

    history: list[dict] = field(default_factory=list)
    """Per-iteration loss / objective records."""

    converged: bool = False
    message: str = ""


# ---------------------------------------------------
# region IEstimator
# ---------------------------------------------------


class MemoryStrategy(Enum):
    """Gradient memory strategy for unrolled long-horizon training.

    Full unrolled backprop stores the graph of every step: O(N) memory
    and exposure to gradient explosion/vanishing. Strategies trade
    compute for memory; they are orthogonal to the loss definition.
    """

    FULL = "full"
    """Store the whole graph. Only for short rollouts / debugging."""

    CHECKPOINT = "checkpoint"
    """Segment the rollout, store only segment boundaries, recompute
    within segments during backward (O(sqrt(N))-style memory)."""

    ADJOINT = "adjoint"
    """O(1) memory in time via adjoint-state integration backwards.
    Research-grade for coupled FVM; reserved, not yet implemented."""


class IEstimator(ABC):
    """Unified interface for parameter inference over estimable targets."""

    @classmethod
    @abstractmethod
    def get_name(cls) -> str:
        """Unique estimator name."""
        pass

    @abstractmethod
    def fit(
        self,
        target: IEstimable,
        data: Any,
        loss: Any,
        **kwargs,
    ) -> EstimationResult:
        """Infer parameters from observations."""
        pass

    @abstractmethod
    def evaluate(
        self,
        target: IEstimable,
        data: Any,
        **kwargs,
    ) -> dict[str, float]:
        """Evaluate current parameters without updating them."""
        pass


# ---------------------------------------------------
# region Optimizer
# ---------------------------------------------------


class Optimizer(IEstimator):
    """End-to-end trainer: unroll solver steps, backprop to theta.

    Skeleton — fixes the contract early. Design constraints encoded here:

    - memory_strategy: CHECKPOINT by default; FULL only for short
      rollouts; ADJOINT reserved.
    - custom VJP: mechanism operators may supply manual backward formulas
      (torch.autograd.Function) for accuracy/memory-critical paths such
      as sparse assembly or WENO reconstruction; the trainer must not
      assume every op is plain-AD.
    - observability: intermediate states are obtained via ISolverCallback
      hooks and DataHub history (in TRAIN mode Samples stay on the graph);
      no training-specific methods are added to ISolver.
    """

    def __init__(
        self,
        optimizer: Any = None,
        memory_strategy: MemoryStrategy = MemoryStrategy.CHECKPOINT,
        unroll_steps: int = 32,
    ):
        self._optimizer = optimizer
        self._memory_strategy = memory_strategy
        self._unroll_steps = unroll_steps

    @classmethod
    def get_name(cls) -> str:
        return "GradientTrainer"

    def fit(self, target: IEstimable, data, loss, **kwargs) -> EstimationResult:
        if not target.supports_gradients():
            raise TypeError(
                f"{type(target).__name__} does not support gradients. "
                "Use Calibrator for black-box targets."
            )
        # TODO(stage-3): unroll loop with the selected MemoryStrategy,
        # attach observation callbacks, loss.backward(), optimizer step.
        raise NotImplementedError(
            "GradientTrainer requires torch-ized operators (roadmap stage 2-3)."
        )

    def evaluate(self, target: IEstimable, data, **kwargs) -> dict[str, float]:
        raise NotImplementedError()


# ---------------------------------------------------
# region Calibrator
# ---------------------------------------------------


class Calibrator(IEstimator):
    """Gradient-free calibrator (DDS).

    Args:
        max_evals: objective evaluation budget.
        r: perturbation size parameter (DDS default 0.2).
        seed: RNG seed for reproducibility.
        x0: optional starting point.
    """

    def __init__(
        self,
        max_evals: int = 5000,
        r: float = 0.2,
        seed: int = None,
        x0: np.ndarray = None,
    ):
        if max_evals < 10:
            raise ValueError("max_evals must be >= 10.")
        if not (0.0 < r < 1.0):
            raise ValueError("DDS perturbation r must be in (0, 1).")
        self._max_evals = int(max_evals)
        self._r = float(r)
        self._rng = np.random.default_rng(seed)
        self._x0 = x0

    @classmethod
    def get_name(cls) -> str:
        return "Calibrator"

    # -- IEstimator -------------------------------------

    def fit(
        self,
        target: IEstimable,
        data: Any,
        loss: Callable[[Any, Any], float],
        names: list[str] = None,
        **kwargs,
    ) -> EstimationResult:
        if not isinstance(target, IEstimable):
            raise TypeError(
                f"{type(target).__name__} is not IEstimable; "
                "mix the estimation contract into the model first."
            )

        names = names or target.param_names()
        lo, hi = target.param_bounds(names)
        d = len(names)
        if d == 0:
            raise ValueError("No estimable parameters selected.")

        # Normalize to unit hypercube: x = lo + z*(hi-lo), z in [0,1]^d
        span = hi - lo
        if np.any(~np.isfinite(span)):
            raise ValueError(
                "DDS requires finite bounds on all selected parameters; "
                f"check param_spec() bounds for {names}."
            )

        z_best = (
            np.clip((np.asarray(self._x0, float) - lo) / span, 0.0, 1.0)
            if self._x0 is not None
            else np.clip((target.get_param_vector(names) - lo) / span, 0.0, 1.0)
        )
        f_best = self._objective(z_best, lo, span, names, target, data, loss)

        result = EstimationResult()
        result.history.append({"eval": 0, "objective": f_best})
        logger.info(f"Calibrator: initial objective={f_best:.6g}, d={d}.")

        m = self._max_evals
        for i in range(1, m):
            # DDS: perturbation probability decays with budget consumption
            prob = 1.0 - np.log(i) / np.log(m)
            dims = self._rng.random(d) < prob
            if not dims.any():
                dims[self._rng.integers(d)] = True

            z_new = z_best.copy()
            z_new[dims] += self._rng.normal(0.0, self._r, size=dims.sum())
            # reflect at bounds
            z_new = np.where(z_new < 0.0, -z_new, z_new)
            z_new = np.where(z_new > 1.0, 2.0 - z_new, z_new)
            z_new = np.clip(z_new, 0.0, 1.0)

            f_new = self._objective(z_new, lo, span, names, target, data, loss)
            if f_new < f_best:
                z_best, f_best = z_new, f_new
                result.history.append({"eval": i, "objective": f_best})

        x_best = lo + z_best * span
        target.set_param_vector(x_best, names)
        target.reset_run()

        result.parameters = dict(zip(names, map(float, x_best)))
        result.converged = True
        result.message = f"DDS finished: {m} evals, best objective={f_best:.6g}."
        logger.info(result.message)
        return result

    def evaluate(self, target: IEstimable, data, loss=None, **kwargs) -> dict:
        target.reset_run()
        sim = target.run()
        if loss is None:
            return {}
        return {"objective": float(loss(sim, data))}

    # -- internals --------------------------------------

    @staticmethod
    def _objective(z, lo, span, names, target, data, loss) -> float:
        target.set_param_vector(lo + z * span, names)
        target.reset_run()
        sim = target.run()
        return float(loss(sim, data))
