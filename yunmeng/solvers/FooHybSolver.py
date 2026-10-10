# -*- encoding: utf-8 -*-
"""
Copyright (C) 2026, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

description
"""

from __future__ import annotations
from dataclasses import asdict, dataclass
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
from yunmeng.interfaces.solver import SolverConfig, SolverMeta, SolverStatus
from yunmeng.solvers.commons import BaseSolver
from yunmeng.operators.FdmDiffusionOp import FdmDiffusionOperator
from yunmeng.operators.NnCorrectionOp import NeuralCorrectionOperator
from yunmeng.numerics.fields import Field
from yunmeng.numerics.linalgs import LinearEqs, assemble_matrix
from yunmeng.numerics.algos import ym_register


@dataclass
class HybridSolverConfig(SolverConfig):
    """Configuration for HybridSolver."""

    start_time: float | str = 0.0
    end_time: float | str = 0.0
    dt: float = 0.01
    device: str = "cpu"

    @classmethod
    def get_solver_name(cls) -> str:
        return "HybridSolver"


@ym_register("solver")
class HybridSolver(BaseSolver):
    """Physics+neural hybrid diffusion solver."""

    @classmethod
    def get_meta(cls) -> SolverMeta:
        return SolverMeta(
            description="Hybrid physics+neural 1D diffusion solver",
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

    def __init__(
        self,
        sid: str,
        mesh: IMesh,
        config: HybridSolverConfig,
        backend: IBackend,
        nu: float = 0.02,
        hidden: int = 32,
    ):
        if not operators:
            operators = [
                FdmDiffusionOperator("fdm_diffusion", "u", nu),
                NeuralCorrectionOperator(
                    "nn_correction",
                    "u",
                    hidden=hidden,
                    device=config.device,
                ),
            ]
        super().__init__(sid, mesh, config, backend)
        self._ops = operators
        self._nu = float(nu)
        self._hidden = int(hidden)

    def _reset_status(self):
        self._status.current_time = self._config.start_time
        self._status.steps = 0
        self._status.time_step = self._config.dt
        self._status.finished = False
        self._status.errors = None

    def step(self, dt: float = None) -> SolverStatus:
        # check if initialized
        if not self._built:
            raise RuntimeError("solver not initialized")

        # prepare for next step
        dt = float(dt if dt is not None else self._config.dt)
        t = self._status.current_time
        field_name = "u"
        field_meta = self._exported_fields[field_name]
        field_size = self._mesh.element_count(field_meta.loc)
        field = self._hub.get_field(field_name)

        # call callbacks
        for cb in self._cbs:
            cb.on_step_begin()

        # evaluate dirchlet boundary conditions
        boundary = self._provider.evaluate(t + dt)
        u = field.values
        for variable, (ids, values) in boundary.constraints.items():
            if variable == field_name:
                u = self._backend.scatter_write(u, ids, values)

        # update field
        field.values = u
        self._hub.touch(field_name)

        # evaluate neumann boundary conditions
        flux_total = self._backend.zeros((field_size,))
        for variable, (ids, values) in boundary.fluxes.items():
            if variable == field_name:
                flux_total = self._backend.scatter_add(
                    flux_total, ids, self._backend.asarray(values)
                )

        # run operators
        implicit_eqs = []
        explicit_total = self._backend.zeros((field_size,))
        for op in self._ops:
            result = op.forward(self._hub, t, dt)
            if result.implicit:
                implicit_eqs.append(result.implicit)
            if result.explicit:
                explicit_total = explicit_total + result.explicit.values

        # assemble and solve linear equations
        if implicit_eqs:
            matrix, rhs = assemble_matrix(implicit_eqs, u)
            rhs = rhs + dt * (explicit_total + flux_total)
            solution = LinearEqs(matrix, Field(field.meta, rhs)).solve()
            field.values = solution.values
        else:
            field.values = u + dt * (explicit_total + flux_total)

        # push field to hub
        self._hub.touch(field_name)
        self._hub.publish(field_name, field, t + dt)

        # update status
        self._status.current_time = t + dt
        self._status.steps += 1
        self._status.time_step = dt
        self._status.finished = self._status.current_time >= self._config.end_time

        # call callbacks
        for cb in self._cbs:
            cb.on_step_end()
        return self._status
