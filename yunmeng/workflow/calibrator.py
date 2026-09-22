# -*- encoding: utf-8 -*-
"""
Copyright (C) 2026, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Estimation layer.
"""

from __future__ import annotations

from typing import Any, Callable
import numpy as np

from yunmeng.interfaces.capabilities import (
    IEstimable,
    EstimationResult,
    IEstimator,
)
from yunmeng.numerics.algos import ym_register
from yunmeng.setting import logger


@ym_register("estimator")
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
