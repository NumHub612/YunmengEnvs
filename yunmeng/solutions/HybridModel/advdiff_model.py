# -*- encoding: utf-8 -*-
"""
AdvDiffModel: the model-layer entry for the hybrid advection-diffusion
solver. Users (and estimators) touch ONLY this class (v1.2 §15).

Implements IEstimable:
- get/set_parameters with namespaced routing (v2.0 §7.2):
      phys.nu / phys.vx / phys.vy   -> mechanism operator parameters
      op.<op_name>.<param>          -> neural operator state_dict entries
- run(n_steps): drives solver.step(); TRAIN mode keeps the graph;
- supports_gradients(): v2.0 §7.4 criteria;
- train()/eval(): recursive mode propagation (v2.0 §6.3).
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import torch

from yunmeng.interfaces import IEstimable, ModelRef, RunMode
from yunmeng.numerics import (
    DataHub,
    Field,
    StaticBoundaryProvider,
    StructuredGrid2D,
)
from yunmeng.solvers import AdvectionDiffusionSolver, AdvDiffConfig

# routing of phys.* names -> (operator name, parameter key)
PHYS_ROUTING = {
    "phys.vx": ("advection", "vx"),
    "phys.vy": ("advection", "vy"),
    "phys.nu": ("diffusion", "nu"),
}


class AdvDiffModel(IEstimable):
    """Hybrid advection-diffusion model (physics ops + optional NN ops)."""

    def __init__(
        self,
        id: str,
        grid: StructuredGrid2D,
        operators: list,
        solver_config: dict | None = None,
        backend: str = "torch",
    ):
        self._id = id
        self._grid = grid
        self._xp = torch if backend == "torch" else np
        self._backend = backend

        cfg = AdvDiffConfig.from_dict(solver_config or {})
        self._solver = AdvectionDiffusionSolver(id, grid, operators, cfg, self._xp)

        # default BC: Dirichlet u = 0 on all boundary nodes (v2.0 §5:
        # boundary as data, held by the model/solver, unseen by operators)
        self._provider = StaticBoundaryProvider(
            {"u": (grid.boundary_node_indices(), 0.0)}, xp=self._xp
        )
        self._mode = RunMode.EVAL
        self._trajectory: list = []

    # -- properties --------------------------------------

    @property
    def id(self) -> str:
        return self._id

    @property
    def grid(self) -> StructuredGrid2D:
        return self._grid

    @property
    def solver(self) -> AdvectionDiffusionSolver:
        return self._solver

    @property
    def mode(self) -> RunMode:
        return self._mode

    # -- setup -------------------------------------------

    def set_initial_condition(self, ic) -> None:
        self._solver.set_initial_condition(ic)

    def initialize(self) -> None:
        self._solver.initialize(self._provider)

    # -- mode (v2.0 §6.3/§8) ------------------------------

    def train(self) -> None:
        # EVAL-only coupling guard: this demo model has no ports/composition,
        # but the check is where is_coupled() would raise RuntimeError.
        self._mode = RunMode.TRAIN
        self._solver.train()

    def eval(self) -> None:
        self._mode = RunMode.EVAL
        self._solver.eval()

    # -- IEstimable: parameters -----------------------------

    def get_parameters(self) -> dict[str, Any]:
        params: dict[str, Any] = {}
        ops = {op.name: op for op in self._solver.operators}
        for pname, (op_name, key) in PHYS_ROUTING.items():
            op = ops.get(op_name)
            if op is not None:
                params[pname] = op.get_parameters()[key]
        for op in self._solver.operators:
            if op.name in ("advection", "diffusion"):
                continue
            for k, v in op.get_parameters().items():
                params[f"op.{op.name}.{k}"] = v
        return params

    def set_parameters(self, params: Mapping[str, Any]) -> None:
        ops = {op.name: op for op in self._solver.operators}
        nn_buffers: dict[str, dict] = {}
        for name, value in params.items():
            if name in PHYS_ROUTING:
                op_name, key = PHYS_ROUTING[name]
                if op_name in ops:
                    ops[op_name].set_parameters({key: value})
            elif name.startswith("op."):
                _, op_name, pkey = name.split(".", 2)
                nn_buffers.setdefault(op_name, {})[pkey] = value
            else:
                raise KeyError(f"Unknown parameter namespace: {name}")
        for op_name, sub in nn_buffers.items():
            if op_name in ops:
                current = ops[op_name].get_parameters()
                current.update(sub)
                ops[op_name].set_parameters(current)

    # -- IEstimable: run --------------------------------------

    def reset_run(self) -> None:
        self._solver.reset()
        self._trajectory = []

    def run(self, n_steps: int, dt: float | None = None) -> torch.Tensor:
        """Unroll n_steps; returns trajectory tensor (n_steps, nx, ny).

        TRAIN: trajectory stays on the autograd graph.
        """
        dt = dt if dt is not None else self._solver.status.time_step
        traj = []
        for _ in range(int(n_steps)):
            self._solver.step(dt)
            traj.append(self._solver.get_solution("u").data)
        self._trajectory = traj
        return torch.stack(traj, dim=0) if self._backend == "torch" else np.stack(traj)

    def supports_gradients(self) -> bool:
        """v2.0 §7.4 criteria: torch backend, all ops differentiable,
        index-written BCs, fixed dt (enforced by config)."""
        if self._backend != "torch":
            return False
        return all(
            getattr(op, "differentiable", False) for op in self._solver.operators
        )
