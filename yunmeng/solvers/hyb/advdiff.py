# -*- encoding: utf-8 -*-
"""
AdvectionDiffusionSolver: explicit FDM solver on a structured 2D grid,
implementing the v2.0 ISolver semantics.

step(dt) internal order (v2.0 §5.3):
    t <- t + dt
    bvals <- provider.evaluate(t)         # boundary evaluated as data
    u     <- scatter-write constraints    # index write, graph-preserving
    hub.publish(boundary, bvals)          # flux products (reserved)
    for op in operators: op.forward(hub, t, dt)
    u     <- u + dt * sum(tendencies)     # explicit Euler advance

Mode propagation (v2.0 §6.3): solver -> DataHub -> IModeSwitchable ops.

TRAIN constraints (v1.3 §20): fixed dt; constraints are index-written
(torch.index_copy: differentiable w.r.t. values, constant indices off
the graph); every step creates fresh tensors (no in-place aliasing).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from yunmeng.interfaces.solver import (
    BoundaryValues,
    OperatorResult,
    SolverStatus,
)
from yunmeng.interfaces.types import RunMode
from yunmeng.numerics.fields import DataHub
from yunmeng.numerics.fields import Field, FieldMeta


@dataclass
class AdvDiffConfig:
    """Configuration for AdvectionDiffusionSolver."""

    time_step: float = 0.02
    end_time: float = 0.4

    @classmethod
    def from_dict(cls, d: dict) -> "AdvDiffConfig":
        keys = {"time_step", "end_time"}
        return cls(**{k: v for k, v in d.items() if k in keys})


class AdvectionDiffusionSolver:
    """Explicit solver for du/dt = -v.grad(u) + nu lap(u) (+ NN terms)."""

    def __init__(self, id: str, grid, operators: list, config: AdvDiffConfig, xp):
        self._id = id
        self._grid = grid
        self._operators = list(operators)
        self._config = config
        self._xp = xp

        self._status = SolverStatus(
            current_time=0.0, end_time=config.end_time, time_step=config.time_step
        )
        self._mode = RunMode.EVAL
        self._hub = DataHub(["u"], mode=self._mode)
        self._provider = None
        self._u: Field | None = None
        self._ic: Field | None = None
        self._built = False

    # -- properties ------------------------------------

    @property
    def id(self) -> str:
        return self._id

    @property
    def status(self) -> SolverStatus:
        return self._status

    @property
    def mode(self) -> RunMode:
        return self._mode

    @property
    def datahub(self) -> DataHub:
        return self._hub

    # -- mode switching (ISolver protocol) ---------------

    def train(self) -> None:
        self._set_mode(RunMode.TRAIN)

    def eval(self) -> None:
        self._set_mode(RunMode.EVAL)

    def _set_mode(self, mode: RunMode) -> None:
        self._mode = mode
        self._hub.set_mode(mode)
        for op in self._operators:
            set_mode = getattr(op, "set_mode", None)  # IModeSwitchable
            if callable(set_mode):
                set_mode(mode)

    # -- assembly ------------------------------------------

    def set_initial_condition(self, ic) -> None:
        """ic: array (nx, ny) or callable(xx, yy) -> array."""
        g = self._grid
        arr = ic(g.coordinates[0], g.coordinates[1]) if callable(ic) else ic
        self._ic = Field.from_array("u", arr, self._xp)

    def get_solution(self, field: str = "u") -> Field:
        return self._u

    @property
    def operators(self) -> list:
        return list(self._operators)

    # -- lifecycle -------------------------------------------

    def initialize(self, boundaries=None) -> None:
        """Cold setup: build operators once per grid, bind provider, apply IC."""
        if self._ic is None:
            raise ValueError(f"Solver {self._id}: no initial condition for 'u'.")
        if not self._built:
            for op in self._operators:
                op.build(self._grid, self._xp)
            self._built = True
        self._provider = boundaries
        self.reset()

    def reset(self) -> None:
        """Cheap rewind to post-initialize state, PRESERVING parameters
        (estimation hot path, v2.0 ISolver.reset contract)."""
        self._hub.clear()
        self._u = Field(self._ic.data, self._ic.meta)  # fresh graph node
        self._hub.publish_field(self._u)
        self._status = SolverStatus(
            current_time=0.0,
            end_time=self._config.end_time,
            time_step=self._config.time_step,
        )

    def step(self, dt: float | None = None) -> SolverStatus:
        if self._u is None:
            raise RuntimeError(f"Solver {self._id}: initialize() not called.")
        dt = float(dt if dt is not None else self._config.time_step)
        t = self._status.current_time + dt

        # 1-2. evaluate provider, scatter-write value constraints
        u = self._u.data
        if self._provider is not None:
            bvals: BoundaryValues = self._provider.evaluate(t)
            u = self._apply_constraints(u, bvals)
            self._hub.publish("boundary", bvals)

        # 3. operator evaluation
        self._hub.publish_field(Field(u, self._u.meta))
        tendency = None
        for op in self._operators:
            res: OperatorResult = op.forward(self._hub, t, dt)
            if res.explicit is not None:
                tendency = (
                    res.explicit.data
                    if tendency is None
                    else tendency + res.explicit.data
                )

        # 4. explicit Euler advance
        u_new = u if tendency is None else u + dt * tendency

        # 5. commit
        self._u = Field(u_new, FieldMeta(name="u", nx=self._grid.nx, ny=self._grid.ny))
        self._hub.publish_field(self._u)

        self._status.current_time = t
        self._status.time_step = dt
        self._status.steps += 1
        if t >= self._status.end_time - 1e-15:
            self._status.finished = True
        return self._status

    # -- internals --------------------------------------------

    def _apply_constraints(self, u, bvals: BoundaryValues):
        """Scatter-write value constraints; differentiable w.r.t. values."""
        cons = bvals.constraints.get("u")
        if cons is None:
            return u
        idx, vals = cons
        if self._xp.__name__ == "torch":
            flat = u.reshape(-1)
            flat = flat.index_copy(0, idx, vals.to(flat.dtype))
            return flat.reshape(u.shape)
        out = np.array(u, copy=True)
        out.reshape(-1)[idx] = vals
        return out
