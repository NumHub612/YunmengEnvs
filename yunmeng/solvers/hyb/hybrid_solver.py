# -*- encoding: utf-8 -*-
"""
Copyright (C) 2026, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

HybridSolver: a physics+neural hybrid solver validating the AI-native path.

Implements:
  - ISolver    : assembly / initialize / step / reset, mode propagation
  - IEstimable : parameter-vector access + reset_run + differentiable run()
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from yunmeng.interfaces.solver.IBoundaryCondition import (
    BoundaryValues,
    IBoundaryCondition,
    IBoundaryProvider,
)
from yunmeng.interfaces.solver.IInitCondition import IInitialCondition
from yunmeng.interfaces.solver.IOperator import (
    IModeSwitchable,
    IOperator,
)
from yunmeng.interfaces.solver.ISolver import (
    IPersistable,
    SolverConfig,
    SolverMeta,
    SolverStatus,
)
from yunmeng.interfaces.solver.ISolverCallback import ISolverCallback
from yunmeng.solvers.hyb.estimation import IEstimable
from yunmeng.interfaces.support.estimable import IParameterized
from yunmeng.interfaces.support.backend import IBackend
from yunmeng.interfaces.support.field import FieldMeta, IField
from yunmeng.interfaces.support.mesh import IMesh
from yunmeng.interfaces.types import (
    ArrayLike,
    ElementType,
    MeshDimension,
    ParamMeta,
    RunMode,
    VariableType,
)
from yunmeng.solvers.hyb.primitives import DataHub, Field

import torch

# ---------------------------------------------------
# region Config / simple boundary provider
# ---------------------------------------------------


@dataclass
class HybridSolverConfig(SolverConfig):
    """Configuration for HybridSolver."""

    dt: float = 0.01
    t0: float = 0.0
    end_time: float = 1.0
    solution_field: str = "u"
    device: str = "cpu"

    @classmethod
    def get_solver_name(cls) -> str:
        return "HybridSolver"


class ListBoundaryProvider:
    """IBoundaryProvider: merges a fixed list of BC rules per evaluation."""

    def __init__(self, bcs: Sequence[IBoundaryCondition] = ()):  # noqa: B008
        self._bcs = list(bcs)

    def add(self, bc: IBoundaryCondition):
        self._bcs.append(bc)

    def evaluate(self, t: float) -> BoundaryValues:
        bv = BoundaryValues(time=t)
        pending: dict[tuple[str, str], list[tuple[ArrayLike, ArrayLike]]] = {}
        for bc in self._bcs:
            channel, values = bc.evaluate(t)
            if channel not in ("constraints", "fluxes", "mixed"):
                raise ValueError(f"unknown BC channel {channel!r}")
            key = (channel, bc.target_field)
            pending.setdefault(key, []).append(
                (np.asarray(bc.region.element_ids, dtype="int64"), np.asarray(values))
            )
        for (channel, var), segments in pending.items():
            ids = np.concatenate([s[0] for s in segments])
            if len(np.unique(ids)) != len(ids):
                raise ValueError(f"overlapping BC segments for field {var!r}")
            vals = np.concatenate([s[1] for s in segments])
            getattr(bv, channel)[var] = (ids, vals)
        return bv


# ---------------------------------------------------
# region HybridSolver
# ---------------------------------------------------


class HybridSolver(IEstimable, IPersistable):
    """Hybrid physics+AI solver over a 1D uniform mesh.

    Step contract (per ISolver):
        evaluate provider -> scatter-write value constraints
        -> publish flux products -> operators' forward -> time advance.
    """

    def __init__(
        self,
        sid: str,
        mesh: IMesh,
        operators: Sequence[IOperator],
        config: HybridSolverConfig,
        backend: IBackend,
    ):
        self._id = sid
        self._mesh = mesh
        self._ops = list(operators)
        self._config = config
        self._backend = backend

        self._mode = RunMode.EVAL
        self._hub = DataHub(mode=self._mode)
        self._ics: dict[str, list[IInitialCondition]] = {}
        self._provider = ListBoundaryProvider()
        self._callbacks: dict[str, ISolverCallback] = {}
        self._equations: list = []

        n = mesh.element_count(ElementType.CELL)
        self._n = n
        self._status = SolverStatus(
            current_time=config.t0,
            end_time=config.end_time,
            time_step=config.dt,
            steps=0,
        )
        self._built = False
        self._ic_values: dict[str, ArrayLike] = {}

    # -- class metadata -----------------------------

    @classmethod
    def get_meta(cls) -> SolverMeta:
        return SolverMeta(
            description="Hybrid physics+neural 1D diffusion solver (validation)",
            kind="hybrid.fdm+nn",
            equation="du/dt = nu * d2u/dx2 + NN(stencil(u))",
            equation_expr="backward-Euler physics + explicit neural correction",
            dimension=MeshDimension.D1,
            fields={"u": FieldMeta(name="u", vtype=VariableType.SCALAR)},
        )

    @classmethod
    def get_name(cls) -> str:
        return "HybridSolver"

    @classmethod
    def get_config_class(cls) -> type[SolverConfig]:
        return HybridSolverConfig

    # -- properties ---------------------------------

    @property
    def id(self) -> str:
        return self._id

    @property
    def status(self) -> SolverStatus:
        return self._status

    @property
    def config(self) -> SolverConfig:
        return self._config

    @property
    def mode(self) -> RunMode:
        return self._mode

    @property
    def datahub(self) -> DataHub:
        return self._hub

    # -- mode switching -----------------------------

    def train(self):
        self._mode = RunMode.TRAIN
        self._hub.set_mode(RunMode.TRAIN)
        for op in self._ops:
            if isinstance(op, IModeSwitchable):
                op.set_mode(RunMode.TRAIN)

    def eval(self):
        self._mode = RunMode.EVAL
        self._hub.set_mode(RunMode.EVAL)
        for op in self._ops:
            if isinstance(op, IModeSwitchable):
                op.set_mode(RunMode.EVAL)

    # -- assembly -----------------------------------

    def set_problems(self, equations: list):
        self._equations = list(equations)

    def get_solution(self, field: str) -> IField:
        return self._hub.get_field(field)

    def add_ic(self, ic: IInitialCondition):
        self._ics.setdefault(ic.target_field, []).append(ic)

    def clear_ics(self, field: str | None = None):
        if field is None:
            self._ics.clear()
        else:
            self._ics.pop(field, None)

    def add_bc(self, bc: IBoundaryCondition):
        self._provider.add(bc)

    def clear_bcs(self, field: str):
        self._provider = ListBoundaryProvider(
            [b for b in self._provider._bcs if b.target_field != field]
        )

    def add_callback(self, cb: ISolverCallback):
        self._callbacks[cb.id] = cb

    def remove_callback(self, cb_id: str):
        self._callbacks.pop(cb_id, None)

    # -- lifecycle ----------------------------------

    def initialize(self, boundaries: IBoundaryProvider | None = None):
        """Cold setup: build operators, allocate the hub, apply ICs."""
        if boundaries is not None:
            self._provider = boundaries
        for op in self._ops:
            op.build(self._mesh, self._backend)
        self._built = True

        self._hub = DataHub(mode=self._mode)
        fname = self._config.solution_field
        meta = FieldMeta(
            name=fname,
            vtype=VariableType.SCALAR,
            loc=ElementType.CELL,
            btype=self._backend.name,
        )
        self._hub.register_field(Field(meta, self._backend.zeros((self._n,))))
        self._apply_ics()
        self._status.current_time = self._config.t0
        self._status.steps = 0
        self._status.finished = False
        self._status.errors = None
        for cb in self._callbacks.values():
            cb.setup(self, self._mesh)

    def _apply_ics(self):
        fname = self._config.solution_field
        field = self._hub.get_field(fname)
        if fname in self._ic_values:
            field.values = self._fresh(self._ic_values[fname])
        for ic in self._ics.get(fname, []):
            ic.apply(field)
        # Normalize to the backend array type and cache for reset().
        field.values = self._backend.asarray(field.values)
        self._ic_values[fname] = self._to_host_copy(field.values)

    def _fresh(self, values: ArrayLike) -> ArrayLike:
        """Fresh leaf array (no stale graph) for IC/reset paths."""
        return self._backend.asarray(self._to_host_copy(values))

    def _to_host_copy(self, values: ArrayLike) -> ArrayLike:
        return np.array(self._backend.to_host(values), copy=True)

    def step(self, dt: float | None = None) -> SolverStatus:
        if not self._built:
            raise RuntimeError("solver not initialized")
        dt = float(dt if dt is not None else self._config.dt)
        t = self._status.current_time
        fname = self._config.solution_field
        field = self._hub.get_field(fname)

        for cb in self._callbacks.values():
            cb.on_step_begin()

        # 1. evaluate provider
        bv = self._provider.evaluate(t + dt)

        # 2. scatter-write value constraints (Dirichlet)
        u = field.values
        for var, (ids, vals) in bv.constraints.items():
            if var != fname:
                continue
            u = self._scatter_write(u, ids, vals)
        field.values = u

        # 3. publish flux products (Neumann) — assembled into rhs below
        flux_contrib = self._backend.zeros((self._n,))
        for var, (ids, vals) in bv.fluxes.items():
            if var != fname:
                continue
            flux_contrib = self._backend.scatter_add(
                flux_contrib, ids, self._backend.asarray(vals)
            )

        # 4. operators' forward
        a_total = None
        rhs_total = None
        explicit_total = self._backend.zeros((self._n,))
        for op in self._ops:
            result = op.forward(self._hub, t, dt)
            if result.implicit is not None:
                a = result.implicit.matrix
                b = result.implicit.rhs
                a_total = a if a_total is None else a_total + a - self._eye()
                rhs_total = b if rhs_total is None else rhs_total + b - u
            if result.explicit is not None:
                explicit_total = explicit_total + result.explicit.values

        # 5. time advance: solve the assembled system
        if a_total is not None:
            rhs = rhs_total + dt * (explicit_total + flux_contrib)
            u_new = self._backend.solve(a_total, rhs)
        else:
            u_new = u + dt * (explicit_total + flux_contrib)
        field.values = u_new

        self._status.current_time = t + dt
        self._status.steps += 1
        self._status.time_step = dt
        self._status.finished = self._status.current_time >= self._config.end_time

        for cb in self._callbacks.values():
            cb.on_step()
            cb.on_step_end()
        return self._status

    def _eye(self) -> ArrayLike:
        return self._backend.asarray(np.eye(self._n))

    def _scatter_write(
        self, u: ArrayLike, ids: ArrayLike, vals: ArrayLike
    ) -> ArrayLike:
        """Functional scatter-write; keeps the autograd graph under torch."""
        vals = self._backend.asarray(vals)
        if isinstance(u, torch.Tensor):
            out = u.clone()
            out[torch.as_tensor(ids, dtype=torch.long, device=u.device)] = vals.to(
                device=u.device, dtype=u.dtype
            )
            return out
        out = np.array(u, copy=True)
        out[np.asarray(ids, dtype="int64")] = np.asarray(vals)
        return out

    def reset(self):
        """Hot-path rewind: keep bindings and current parameters, restore ICs."""
        fname = self._config.solution_field
        self._hub.clear_products()
        field = self._hub.get_field(fname)
        field.values = self._fresh(self._ic_values[fname])
        self._status.current_time = self._config.t0
        self._status.steps = 0
        self._status.finished = False
        self._status.errors = None

    # -- IEstimable ---------------------------------

    def _param_ops(self) -> list[tuple[int, IParameterized]]:
        return [
            (i, op) for i, op in enumerate(self._ops) if isinstance(op, IParameterized)
        ]

    def param_spec(self) -> list[ParamMeta]:
        spec = []
        for i, op in self._param_ops():
            for name, val in op.get_parameters().items():
                size = int(np.prod(self._to_host_copy(val).shape))
                spec.append(
                    ParamMeta(
                        name=f"op{i}.{name}",
                        description=f"{op.get_name()}",
                        default=0.0,
                    )
                )
        return spec

    def param_names(self) -> list[str]:
        return [p.name for p in self.param_spec()]

    def get_param_vector(self, names: list[str] | None = None) -> ArrayLike:
        params = {}
        for i, op in self._param_ops():
            for name, val in op.get_parameters().items():
                params[f"op{i}.{name}"] = self._to_host_copy(val).ravel()
        names = names or list(params.keys())
        return np.concatenate([params[n] for n in names])

    def set_param_vector(self, values: ArrayLike, names: list[str] | None = None):
        values = np.asarray(values, dtype="float64")
        grouped: dict[int, dict[str, ArrayLike]] = {}
        cursor = 0
        all_names = self.param_names()
        names = names or all_names
        shapes = {}
        for i, op in self._param_ops():
            for name, val in op.get_parameters().items():
                shapes[f"op{i}.{name}"] = self._to_host_copy(val).shape
        for n in names:
            size = int(np.prod(shapes[n]))
            head, _, tail = n.partition(".")
            op_idx = int(head[2:])
            grouped.setdefault(op_idx, {})[tail] = values[
                cursor : cursor + size
            ].reshape(shapes[n])
            cursor += size
        for op_idx, params in grouped.items():
            self._ops[op_idx].set_parameters(params)

    def reset_run(self):
        self.reset()

    def run(self, n_steps: int, **kwargs) -> Any:
        """Forward rollout of n_steps. Under TRAIN the returned trajectory
        stays on the autograd graph."""
        traj = []
        for _ in range(int(n_steps)):
            self.step()
            u = self._hub.get_field(self._config.solution_field).values
            traj.append(u)
        if isinstance(traj[0], torch.Tensor):
            return torch.stack(traj)
        return np.stack(traj)

    def supports_gradients(self) -> bool:
        return self._backend.differentiable

    # -- ISnapshotable (minimal; θ excluded by design) ---

    def save(self, path: str):
        import json

        payload = {
            "id": self._id,
            "config": self._config.to_dict(),
            "time": self._status.current_time,
            "ic": {k: v.tolist() for k, v in self._ic_values.items()},
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> "HybridSolver":
        raise NotImplementedError(
            "snapshot load requires mesh/operators/backend resolution; "
            "rebuild via constructor and re-register parameters from the "
            "model artifact store."
        )
