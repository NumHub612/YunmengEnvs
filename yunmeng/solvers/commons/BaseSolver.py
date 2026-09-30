# -*- encoding: utf-8 -*-
"""
Copyright (C) 2026, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Basic solver implementation.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any, Tuple
import numpy as np

from yunmeng.interfaces.capabilities import (
    IDifferentiable,
    IEstimable,
    IParameterized,
    ParamMeta,
    propagate_mode,
)
from yunmeng.interfaces.solver import (
    BoundaryValues,
    IBoundaryCondition,
    IBoundaryProvider,
    IInitialCondition,
    IOperator,
    ISolverCallback,
    ISolver,
    SolverConfig,
    SolverMeta,
    SolverStatus,
    IEquation,
)
from yunmeng.interfaces.supports import (
    FieldMeta,
    IBackend,
    IField,
    IMesh,
    Region,
)
from yunmeng.interfaces.types import (
    ArrayLike,
    ElementType,
    MeshDimension,
    RunMode,
    VariableType,
)

from yunmeng.solvers.commons.boundaries import ListBoundaryProvider
from yunmeng.numerics.fields import DataHub, Field
from yunmeng.numerics.linalgs import LinearEqs, NumpyMatrix, TorchMatrix
from yunmeng.setting import logger


@dataclass
class BaseSolverConfig(SolverConfig):
    """Configuration shared."""

    start_time: float | str = 0.0
    end_time: float | str = 0.0
    dt: float = 0.01


class BaseSolver(ISolver, IEstimable, IDifferentiable):
    """Shared one-dimensional operator-stepping implementation."""

    def __init__(self, sid: str, mesh: IMesh, config: SolverConfig, backend: IBackend):
        self._id = sid
        self._mesh: IMesh = mesh
        self._config: SolverConfig = config
        self._backend: IBackend = backend
        self._mode = RunMode.EVAL
        self._status = SolverStatus()
        self._hub = DataHub(mode=self._mode, mesh=mesh)

        self._ops: list[IOperator] = []
        self._ics: list[IInitialCondition] = []
        self._bcs: list[IBoundaryCondition] = []
        self._cbs: list[ISolverCallback] = []
        self._eqs: list[IEquation] = []

        self._provider = ListBoundaryProvider()
        self._built = False
        self._eq_fixed = True
        self._init_values: dict[str, ArrayLike] = {}
        self._exported_fields = self.get_meta().fields

    # region properties -----------------------------

    @property
    def id(self) -> str:
        return self._id

    @property
    def status(self) -> SolverStatus:
        return self._status

    @property
    def ics(self) -> list[IInitialCondition]:
        return self._ics

    @property
    def bcs(self) -> list[IBoundaryCondition]:
        return self._provider.rules

    @property
    def ops(self) -> list[IOperator]:
        return self._ops

    @property
    def mode(self) -> RunMode:
        return self._mode

    # region mode -----------------------------------

    def train(self):
        self._mode = RunMode.TRAIN
        self._hub.set_mode(RunMode.TRAIN)
        propagate_mode(RunMode.TRAIN, self._ops)

    def eval(self):
        self._mode = RunMode.EVAL
        self._hub.set_mode(RunMode.EVAL)
        propagate_mode(RunMode.EVAL, self._ops)

    # region problem --------------------------------

    def set_problems(self, equations: list[IEquation]):
        if self._eq_fixed:
            raise RuntimeError("Cannot reset equations.")
        self._eqs = equations

    def get_solution(self, field: str) -> IField:
        return self._hub.get_field(field)

    # region assembly -------------------------------

    def add_callback(self, cb: ISolverCallback):
        if self._index(self._cbs, cb):
            raise RuntimeError(f"Callback {cb.id} already exists.")
        self._cbs.append(cb)

    def remove_callback(self, cb_id: str):
        index = self._index(self._cbs, cb_id)
        if index:
            del self._cbs[index]

    def add_op(self, op: IOperator):
        if self._index(self._ops, op):
            raise RuntimeError(f"Operator {op.id} already exists.")
        self._ops.append(op)

    def remove_op(self, op_id: str):
        index = self._index(self._ops, op_id)
        if index:
            del self._ops[index]

    def add_ic(self, ic: IInitialCondition):
        if self._index(self._ics, ic):
            raise RuntimeError(f"IC {ic.id} already exists.")
        self._ics.append(ic)

    def remove_ic(self, ic_id: str):
        index = self._index(self._ics, ic_id)
        if index:
            del self._ics[index]

    def add_bc(self, bc: IBoundaryCondition):
        if self._index(self._bcs, bc):
            raise RuntimeError(f"BC {bc.id} already exists.")
        self._bcs.append(bc)

    def remove_bc(self, bc_id):
        index = self._index(self._bcs, bc_id)
        if index:
            del self._bcs[index]

    def _index(self, sets: list, obj: object) -> int:
        ids = [s.id for s in sets]
        if obj.id not in ids:
            return None
        return ids.index(obj.id)

    # region lifecycle ------------------------------

    def initialize(self):
        self._provider = ListBoundaryProvider(self._bcs)
        for op in self._ops:
            op.build(self._mesh, self._backend)

        self._hub = DataHub(mode=self._mode, mesh=self._mesh)
        self._registry_fields()
        self._apply_ics()
        self._reset_status()
        self._built = True

        for cb in self._cbs:
            cb.setup(self, self._mesh)

    def _registry_fields(self):
        """Registry fields to datahub."""
        for field_name, meta in self._exported_fields.items():
            size = self._mesh.element_count(meta.loc)
            field = Field(meta, self._backend.zeros((size,)))
            self._hub.register_field(field)

    def _apply_ics(self):
        """Apply initial conditions."""
        for field_name in self._exported_fields.keys():
            field = self._hub.get_field(field_name)
            # reset initial values
            if field_name in self._init_values:
                field.values = self._fresh(self._init_values[field_name])
            # apply initial condition
            idx = self._index(self._ics, field_name)
            if idx is not None:
                self._ics[idx].apply(field)
            # backend normalization
            field.values = self._backend.asarray(field.values)
            # recache field
            self._init_values[field_name] = self._host_copy(field.values)
            # touch field
            self._hub.touch(field_name)

    def _fresh(self, values: ArrayLike) -> ArrayLike:
        """Fresh values for each step."""
        return self._backend.asarray(self._host_copy(values))

    def _host_copy(self, values: ArrayLike) -> np.ndarray:
        """Copy values to host."""
        return np.array(self._backend.to_host(values), copy=True)

    def step(self, dt: float = None) -> SolverStatus:
        raise NotImplementedError

    def reset(self):
        for field_name in self._exported_fields.keys():
            self._hub.clear_products()
            field = self._hub.get_field(field_name)
            field.values = self._fresh(self._init_values[field_name])
            self._hub.touch(field_name)
        self._reset_status()

    def _reset_status(self):
        """Reset solver status."""
        raise NotImplementedError

    # region Parameterized --------------------------

    def _parameter_ops(self) -> list[Tuple[int, IParameterized]]:
        return [
            (index, op)
            for index, op in enumerate(self._ops)
            if isinstance(op, IParameterized)
        ]

    def parameter_metas(self) -> list[ParamMeta]:
        metas = []
        for index, op in self._parameter_ops():
            for meta in op.parameter_metas():
                metas.append(
                    ParamMeta(
                        name=f"op{index}.{meta.name}",
                        description=meta.description,
                        dtype=meta.dtype,
                        bounds=meta.bounds,
                        default=meta.default,
                        required=meta.required,
                    )
                )
        return metas

    def get_parameters(self, names: list[str] = None) -> ArrayLike:
        vectors = {}
        for index, op in self._parameter_ops():
            local_names = op.parameter_names()
            values = np.asarray(op.get_parameters(local_names), dtype="float64")
            for local, value in zip(local_names, values):
                vectors[f"op{index}.{local}"] = value
        names = names or list(vectors)
        return np.asarray([vectors[name] for name in names], dtype="float64")

    def set_parameters(self, values: ArrayLike, names: list[str] = None):
        names = names or self.parameter_names()
        values = np.asarray(values, dtype="float64").reshape(-1)
        if len(values) != len(names):
            raise ValueError(f"Expected {len(names)} values, got {len(values)}")

        grouped = {}
        for name, value in zip(names, values):
            head, _, local = name.partition(".")
            if not head.startswith("op"):
                raise ValueError(f"invalid parameter namespace {name!r}")
            index = int(head[2:])
            grouped.setdefault(index, []).append((local, value))

        for index, pairs in grouped.items():
            op = self._ops[index]
            op.set_parameters(
                np.asarray([value for _, value in pairs]),
                [name for name, _ in pairs],
            )

    def parameter_bounds(self, names: list[str] = None) -> tuple:
        names = names or self.parameter_names()
        lower, upper = [], []
        grouped = {}
        for name in names:
            head, _, local = name.partition(".")
            grouped.setdefault(int(head[2:]), []).append(local)
        for index, local_names in grouped.items():
            lo, hi = self._ops[index].parameter_bounds(local_names)
            lower.extend(lo)
            upper.extend(hi)
        return lower, upper

    # region Differentiable -------------------------

    def grad_parameters(self) -> list[ArrayLike]:
        leaves = []
        for op in self._ops:
            if isinstance(op, IDifferentiable):
                leaves.extend(op.grad_parameters())
        return leaves

    # region Estimable ------------------------------

    def run(self, n_steps: int) -> Any:
        trajectory = []
        for _ in range(int(n_steps)):
            self.step()
            for field_name in self._exported_fields.keys():
                trajectory.append(self.get_solution(field_name).values)
        if not trajectory:
            raise ValueError("n_steps must be positive")
        return self._backend.stack(trajectory, axis=0)

    def supports_gradients(self) -> bool:
        return self._backend.differentiable and all(
            getattr(op, "differentiable", True) for op in self._ops
        )

    # region Snapshottable --------------------------

    def snapshot(self):
        return {
            "id": self._id,
            "config": self._config.to_dict(),
            "status": asdict(self._status),
            "parameters": self.get_parameters(),
            "initial_conditions": {
                name: self._host_copy(values)
                for name, values in self._init_values.items()
            },
            "solutions": {
                field_name: self._host_copy(self.get_solution(field_name).values)
                for field_name in self._exported_fields.keys()
            },
        }

    def restore(self, snapshot):
        self.set_parameters(snapshot["parameters"])
        self._init_values = {
            name: np.asarray(values, dtype="float64")
            for name, values in snapshot["initial_conditions"].items()
        }
        for name, value in snapshot["status"].items():
            setattr(self._status, name, value)
        for field_name in self._exported_fields.keys():
            field = self.get_solution(field_name)
            field.values = self._fresh(snapshot["solutions"][field_name])

    def save(self, path: str):
        payload = self.snapshot()
        payload["parameters"] = payload["parameters"].tolist()
        payload["initial_conditions"] = {
            name: values.tolist()
            for name, values in payload["initial_conditions"].items()
        }
        payload["solutions"] = {
            field_name: values.tolist()
            for field_name, values in payload["solutions"].items()
        }
        with open(path, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> BaseSolver:
        raise NotImplementedError
