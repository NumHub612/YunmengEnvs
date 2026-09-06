# -*- encoding: utf-8 -*-
"""
Boundary providers (v2.0 §5): BC evaluated as data, per instant.

- StaticBoundaryProvider: constant value constraints (e.g. Dirichlet 0).
- SeriesBoundaryProvider: time-series driven values with linear interp
  (driver for training-data reinjection).

Providers are owned by model/solver; operators never see them.
"""

from __future__ import annotations

from typing import Callable

from yunmeng.interfaces.solver import BoundaryValues


class StaticBoundaryProvider:
    """Constant per-variable boundary values.

    bindings: {var: (node_indices, value_or_callable_of_t)}
    """

    def __init__(self, bindings: dict[str, tuple], xp):
        self._bindings = bindings
        self._xp = xp

    def evaluate(self, t: float) -> BoundaryValues:
        xp = self._xp
        constraints = {}
        for var, (idx, value) in self._bindings.items():
            v = value(t) if isinstance(value, Callable) else value
            if xp.__name__ == "torch":
                import torch

                idx_t = torch.as_tensor(idx, dtype=torch.long)
                vals = torch.as_tensor(v, dtype=torch.float64)
                vals = vals.expand(len(idx)) if vals.dim() == 0 else vals
            else:
                import numpy as np

                idx_t = np.asarray(idx, dtype=np.int64)
                vals = np.full(len(idx), float(v), dtype=np.float64)
            constraints[var] = (idx_t, vals)
        return BoundaryValues(time=t, constraints=constraints)


class SeriesBoundaryProvider:
    """Time-series driven boundary values (linear interpolation in time).

    bindings: {var: (node_indices, times, values)}; values vary per instant,
    shared across the variable's node set (uniform segment value).
    """

    def __init__(self, bindings: dict[str, tuple], xp):
        self._bindings = bindings
        self._xp = xp

    def evaluate(self, t: float) -> BoundaryValues:
        xp = self._xp
        constraints = {}
        for var, (idx, times, values) in self._bindings.items():
            # piecewise-linear interpolation at time t (float math: t is
            # not on the graph; values may be, via as_tensor below)
            times = list(times)
            values = list(values)
            if t <= times[0]:
                v = values[0]
            elif t >= times[-1]:
                v = values[-1]
            else:
                k = max(i for i, ti in enumerate(times) if ti <= t)
                w = (t - times[k]) / (times[k + 1] - times[k])
                v = values[k] + w * (values[k + 1] - values[k])
            if xp.__name__ == "torch":
                import torch

                idx_t = torch.as_tensor(idx, dtype=torch.long)
                vals = torch.as_tensor(float(v), dtype=torch.float64).expand(len(idx))
            else:
                import numpy as np

                idx_t = np.asarray(idx, dtype=np.int64)
                vals = np.full(len(idx), float(v), dtype=np.float64)
            constraints[var] = (idx_t, vals)
        return BoundaryValues(time=t, constraints=constraints)
