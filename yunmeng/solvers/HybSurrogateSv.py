# -*- encoding: utf-8 -*-
"""
Copyright (C) 2026, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

HybLapSolver: diffusion solver driven by an AI surrogate Laplacian.

    du/dt = NN(stencil(u))          (explicit Euler stepping)

The surrogate fully replaces the discretized Laplacian. For safety in
online operation the solver keeps a DORMANT physics operator (FVM
diffusion, assembled on the same mesh/backend at initialize time);
when an external monitor reports the surrogate error over threshold,
`fallback_to_physics()` swaps the operator list in place, and
`resume_surrogate()` swaps it back once the surrogate has been
retrained. The swap is hot: ICs, boundary bindings and DataHub are
untouched, the current field state carries over.
"""

from __future__ import annotations

from dataclasses import dataclass

from yunmeng.standard.solver import SolverConfig, SolverMeta, SolverStatus
from yunmeng.standard.supports import FieldMeta, IBackend, IMesh, Sample
from yunmeng.standard.types import ElementType, MeshDimension, VariableType
from yunmeng.standard.capabilities import propagate_mode
from yunmeng.solvers.commons import BaseSolver
from yunmeng.numerics.fields import Field
from yunmeng.numerics.linalgs import LinearEqs, assemble_matrix
from yunmeng.operators.FvmDiffusionOp import FvmDiffusionOperator
from yunmeng.operators.NnDiffusionOp import NeuralDiffusionOperator


@dataclass
class HybSurrogateSolverConfig(SolverConfig):
    """Configuration for HybLapSolver."""

    start_time: float | str = 0.0
    end_time: float | str = 0.0
    dt: float = 0.01
    nu: float = 0.02
    hidden: int = 32
    device: str = "cpu"

    @classmethod
    def get_solver_name(cls) -> str:
        return "HybSurrogateSolver"


class HybSurrogateSolver(BaseSolver):
    """AI surrogate-Laplacian diffusion solver with physics fallback."""

    @classmethod
    def get_meta(cls) -> SolverMeta:
        return SolverMeta(
            description="Surrogate-Laplacian 1D diffusion solver, physics fallback",
            kind="hybrid.nn.laplacian",
            equation="du/dt = NN(stencil(u))",
            equation_expr="explicit surrogate Laplacian, dormant FVM fallback",
            dimension=MeshDimension.D1,
            fields={
                "u": FieldMeta(
                    name="u", vtype=VariableType.SCALAR, loc=ElementType.CELL
                )
            },
        )

    @classmethod
    def get_name(cls) -> str:
        return "HybLapSolver"

    @classmethod
    def get_config_class(cls) -> type[SolverConfig]:
        return HybSurrogateSolverConfig

    def __init__(
        self,
        sid: str,
        mesh: IMesh,
        config: HybSurrogateSolverConfig,
        backend: IBackend,
    ):
        super().__init__(sid, mesh, config, backend)

        # dormant physics operator for fallback
        self._physics_op = FvmDiffusionOperator("fvm", "u", self._config.nu)
        self._surrogate_op = None
        self._fallback_active = False

    @property
    def fallback_active(self) -> bool:
        return self._fallback_active

    def _reset_status(self):
        self._status.current_time = self._config.start_time
        self._status.steps = 0
        self._status.time_step = self._config.dt
        self._status.finished = False
        self._status.errors = None

    def initialize(self):
        if not self._ops:
            self._ops = [
                NeuralDiffusionOperator(
                    "nn",
                    "u",
                    hidden=self._config.hidden,
                    device=self._config.device,
                    nu_hint=self._config.nu,
                )
            ]
        self._surrogate_op = self._ops[0]

        super().initialize()

        # build the dormant physics operator so the fallback swap is hot
        self._physics_op.build(self._mesh, self._backend)

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
        field = self._hub.field(field_name)

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
            # call operator
            result = op.forward(self._hub, t, dt)
            if result.implicit:
                implicit_eqs.append(result.implicit)
            if result.explicit:
                explicit_total = explicit_total + result.explicit.values

            # cache operator produces
            declared = {dp.name: dp for dp in op.produces()}
            for key, value in (result.products or {}).items():
                dp = declared[key]
                sample = Sample(value, t + dt, self._hub.version(...))
                deps = [c.field for c in op.consumes()]
                self._hub.cache(dp, sample, deps)

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
        self._hub.commit(field, t + dt)

        # update status
        self._status.current_time = t + dt
        self._status.steps += 1
        self._status.time_step = dt
        self._status.finished = self._status.current_time >= self._config.end_time

        # call callbacks
        for cb in self._cbs:
            cb.on_step_end()
        return self._status

    def fallback_to_physics(self):
        """Swap the surrogate out, dormant FVM physics in. Hot: the current
        field state, ICs and boundary bindings are preserved."""
        if self._fallback_active:
            return
        self._ops = [self._physics_op]
        self._fallback_active = True
        propagate_mode(self._mode, self._ops)

    def resume_surrogate(self):
        """Swap the (typically retrained) surrogate back in."""
        if not self._fallback_active:
            return
        self._ops = [self._surrogate_op]
        self._fallback_active = False
        propagate_mode(self._mode, self._ops)

    # -- θ / gradient channels ----------------------
    #
    # Always bound to the surrogate operator, independent of which
    # operator currently drives the solution. Otherwise parameter
    # access and gradient training would silently break while the
    # physics fallback is active.

    def parameter_metas(self):
        return self._surrogate_op.parameter_metas()

    def parameter_names(self):
        return self._surrogate_op.parameter_names()

    def get_parameters(self, names=None):
        return self._surrogate_op.get_parameters(names)

    def set_parameters(self, values, names=None):
        names = names or self.parameter_names()
        local = [n.partition(".")[2] if n.startswith("op") else n for n in names]
        self._surrogate_op.set_parameters(values, local)

    def parameter_bounds(self, names=None):
        return self._surrogate_op.parameter_bounds(names)

    def grad_parameters(self):
        return self._surrogate_op.grad_parameters()
