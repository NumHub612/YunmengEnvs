# -*- encoding: utf-8 -*-
"""
Copyright (C) 2026, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

description
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any, Sequence
import numpy as np
import torch

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


@dataclass
class FdmSolverConfig(SolverConfig):
    """Configuration shared by pure-FDM and hybrid diffusion solvers."""

    dt: float = 0.01
    t0: float = 0.0
    end_time: float = 1.0
    solution_field: str = "u"
    device: str = "cpu"

    @classmethod
    def get_solver_name(cls) -> str:
        return "FdmSolver"


@dataclass
class HybridSolverConfig(FdmSolverConfig):
    """Configuration for HybridSolver."""

    @classmethod
    def get_solver_name(cls) -> str:
        return "HybridSolver"


class BaseSolver(ISolver, IEstimable, IDifferentiable):
    """Shared one-dimensional operator-stepping implementation."""

    def __init__(
        self,
        sid: str,
        mesh: IMesh,
        operators: Sequence,
        config: FdmSolverConfig,
        backend: IBackend,
    ):
        self._id = sid
        self._mesh = mesh
        self._ops: list[IOperator] = list(operators)
        self._config = config
        self._backend = backend
        self._mode = RunMode.EVAL
        self._n = mesh.element_count(ElementType.CELL)

        self._hub = DataHub(mode=self._mode, mesh=mesh)
        self._ics = {}
        self._provider = ListBoundaryProvider()
        self._callbacks: dict[str, ISolverCallback] = {}
        self._equations = []
        self._built = False
        self._ic_values = {}
        self._status = SolverStatus(
            current_time=config.t0,
            end_time=config.end_time,
            time_step=config.dt,
            steps=0,
        )

    @classmethod
    def get_meta(cls) -> SolverMeta:
        return SolverMeta(
            description="One-dimensional operator stepping solver",
            kind="fdm.operator-composition",
            equation="operator-defined",
            equation_expr="implicit operators + explicit operators",
            dimension=MeshDimension.D1,
            fields={"u": FieldMeta(name="u", vtype=VariableType.SCALAR)},
        )

    @classmethod
    def get_name(cls) -> str:
        return "SteppingSolver"

    @classmethod
    def get_config_class(cls) -> type[SolverConfig]:
        return FdmSolverConfig

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

    def train(self):
        self._mode = RunMode.TRAIN
        self._hub.set_mode(RunMode.TRAIN)
        propagate_mode(RunMode.TRAIN, self._ops)

    def eval(self):
        self._mode = RunMode.EVAL
        self._hub.set_mode(RunMode.EVAL)
        propagate_mode(RunMode.EVAL, self._ops)

    def set_problems(self, equations: list):
        self._equations = list(equations)

    def get_solution(self, field: str) -> IField:
        return self._hub.get_field(field)

    def add_ic(self, ic):
        self._ics.setdefault(ic.target_field, []).append(ic)

    def clear_ics(self, field: str = None):
        if field is None:
            self._ics.clear()
        else:
            self._ics.pop(field, None)

    def add_bc(self, bc: IBoundaryCondition):
        self._provider.add(bc)

    def clear_bcs(self, field: str):
        self._provider = ListBoundaryProvider(
            [bc for bc in self._provider.rules if bc.target_field != field]
        )

    def add_callback(self, cb):
        self._callbacks[cb.id] = cb

    def remove_callback(self, cb_id: str):
        self._callbacks.pop(cb_id, None)

    def initialize(self, boundaries: IBoundaryProvider = None):
        if boundaries is not None:
            self._provider = boundaries
        for op in self._ops:
            op.build(self._mesh, self._backend)
        self._built = True

        self._hub = DataHub(mode=self._mode, mesh=self._mesh)
        field_name = self._config.solution_field
        meta = FieldMeta(
            name=field_name,
            vtype=VariableType.SCALAR,
            loc=ElementType.CELL,
            btype=self._backend.name,
            device=self._config.device,
        )
        self._hub.register_field(Field(meta, self._backend.zeros((self._n,))))
        self._apply_ics()
        self._reset_status()
        for cb in self._callbacks.values():
            cb.setup(self, self._mesh)

    def _apply_ics(self):
        field_name = self._config.solution_field
        field = self._hub.get_field(field_name)
        if field_name in self._ic_values:
            field.values = self._fresh(self._ic_values[field_name])
        for ic in self._ics.get(field_name, []):
            ic.apply(field)
        field.values = self._backend.asarray(field.values)
        self._ic_values[field_name] = self._host_copy(field.values)
        self._hub.touch(field_name)

    def _fresh(self, values: ArrayLike) -> ArrayLike:
        return self._backend.asarray(self._host_copy(values))

    def _host_copy(self, values: ArrayLike) -> np.ndarray:
        return np.array(self._backend.to_host(values), copy=True)

    def step(self, dt: float = None) -> SolverStatus:
        if not self._built:
            raise RuntimeError("solver not initialized")
        dt = float(dt if dt is not None else self._config.dt)
        t = self._status.current_time
        field_name = self._config.solution_field
        field = self._hub.get_field(field_name)

        for cb in self._callbacks.values():
            cb.on_step_begin()

        boundary = self._provider.evaluate(t + dt)
        u = field.values
        for variable, (ids, values) in boundary.constraints.items():
            if variable == field_name:
                u = self._scatter_write(u, ids, values)
        field.values = u
        self._hub.touch(field_name)

        flux_total = self._backend.zeros((self._n,))
        for variable, (ids, values) in boundary.fluxes.items():
            if variable == field_name:
                flux_total = self._backend.scatter_add(
                    flux_total, ids, self._backend.asarray(values)
                )

        implicit_eqs = []
        explicit_total = self._backend.zeros((self._n,))
        for op in self._ops:
            result = op.forward(self._hub, t, dt)
            if result.implicit is not None:
                implicit_eqs.append(result.implicit)
            if result.explicit is not None:
                explicit_total = explicit_total + result.explicit.values

        if implicit_eqs:
            matrix, rhs = self._assemble_implicit(implicit_eqs, u)
            rhs = rhs + dt * (explicit_total + flux_total)
            solution = LinearEqs(matrix, Field(field.meta, rhs)).solve()
            field.values = solution.values
        else:
            field.values = u + dt * (explicit_total + flux_total)

        self._hub.touch(field_name)
        self._hub.publish(field_name, field, t + dt)

        self._status.current_time = t + dt
        self._status.steps += 1
        self._status.time_step = dt
        self._status.finished = self._status.current_time >= self._config.end_time

        for cb in self._callbacks.values():
            cb.on_step()
            cb.on_step_end()
        return self._status

    def _assemble_implicit(self, equations: list[LinearEqs], u: ArrayLike):
        matrix = None
        rhs = None
        for eqs in equations:
            if matrix is None:
                matrix = eqs.matrix
                rhs = eqs.rhs.values
            else:
                matrix = matrix + eqs.matrix - self._identity_matrix()
                rhs = rhs + eqs.rhs.values - u
        return matrix, rhs

    def _identity_matrix(self):
        if self._backend.name == "torch":
            return TorchMatrix.identity(self._n, device=str(self._backend.device))
        return NumpyMatrix.identity(self._n)

    def _scatter_write(self, u: ArrayLike, ids: ArrayLike, values: ArrayLike):
        values = self._backend.asarray(values)
        if isinstance(u, torch.Tensor):
            result = u.clone()
            index = torch.as_tensor(ids, dtype=torch.long, device=u.device)
            result[index] = values.to(device=u.device, dtype=u.dtype)
            return result
        result = np.array(u, copy=True)
        result[np.asarray(ids, dtype="int64")] = np.asarray(values)
        return result

    def reset(self):
        field_name = self._config.solution_field
        self._hub.clear_products()
        field = self._hub.get_field(field_name)
        field.values = self._fresh(self._ic_values[field_name])
        self._hub.touch(field_name)
        self._reset_status()

    def _reset_status(self):
        self._status.current_time = self._config.t0
        self._status.steps = 0
        self._status.time_step = self._config.dt
        self._status.finished = False
        self._status.errors = None

    def _parameter_ops(self) -> list[int, IParameterized]:
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

    def grad_parameters(self) -> list[ArrayLike]:
        leaves = []
        for op in self._ops:
            if isinstance(op, IDifferentiable):
                leaves.extend(op.grad_parameters())
        return leaves

    def run(self, n_steps: int, **kwargs) -> Any:
        trajectory = []
        for _ in range(int(n_steps)):
            self.step()
            trajectory.append(self.get_solution(self._config.solution_field).values)
        if not trajectory:
            raise ValueError("n_steps must be positive")
        return self._backend.stack(trajectory, axis=0)

    def supports_gradients(self) -> bool:
        return self._backend.differentiable and all(
            getattr(op, "differentiable", True) for op in self._ops
        )

    def snapshot(self):
        return {
            "id": self._id,
            "config": self._config.to_dict(),
            "status": asdict(self._status),
            "parameters": self.get_parameters(),
            "initial_conditions": {
                name: self._host_copy(values)
                for name, values in self._ic_values.items()
            },
            "solution": self._host_copy(
                self.get_solution(self._config.solution_field).values
            ),
        }

    def restore(self, snapshot):
        self.set_parameters(snapshot["parameters"])
        self._ic_values = {
            name: np.asarray(values, dtype="float64")
            for name, values in snapshot["initial_conditions"].items()
        }
        field = self.get_solution(self._config.solution_field)
        field.values = self._fresh(snapshot["solution"])
        for name, value in snapshot["status"].items():
            setattr(self._status, name, value)

    def save(self, path: str):
        payload = self.snapshot()
        payload["parameters"] = payload["parameters"].tolist()
        payload["initial_conditions"] = {
            name: values.tolist()
            for name, values in payload["initial_conditions"].items()
        }
        payload["solution"] = payload["solution"].tolist()
        with open(path, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str):
        raise NotImplementedError(
            "snapshot load requires mesh, operators and backend; rebuild via "
            "the constructor, then call restore() with the saved payload"
        )
