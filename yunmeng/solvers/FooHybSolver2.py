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
from typing import Sequence

from yunmeng.interfaces.solver import SolverConfig, SolverMeta
from yunmeng.interfaces.supports import FieldMeta, IBackend, IMesh
from yunmeng.interfaces.types import MeshDimension, VariableType
from yunmeng.solvers.commons import BaseSolver
from yunmeng.operators.FvmDiffusionOp import FvmDiffusionOperator
from yunmeng.operators.NnLaplacianOp import NeuralLaplacianOperator


@dataclass
class HybLapSolverConfig(SolverConfig):
    """Configuration for HybLapSolver."""

    dt: float = 0.01
    t0: float = 0.0
    end_time: float = 1.0
    solution_field: str = "u"
    device: str = "cpu"

    @classmethod
    def get_solver_name(cls) -> str:
        return "HybLapSolver"


class HybLapSolver(BaseSolver):
    """AI surrogate-Laplacian diffusion solver with physics fallback."""

    def __init__(
        self,
        sid: str,
        mesh: IMesh,
        config: HybLapSolverConfig,
        backend: IBackend,
        nu: float = 0.02,
        hidden: int = 32,
        operators: Sequence = None,
    ):
        self._surrogate_op = NeuralLaplacianOperator(
            config.solution_field,
            hidden=hidden,
            device=config.device,
        )
        # dormant physics operator for fallback; built in initialize()
        self._physics_op = FvmDiffusionOperator(config.solution_field, nu)
        if operators is None:
            operators = [self._surrogate_op]
        super().__init__(sid, mesh, operators, config, backend)
        self._nu = float(nu)
        self._hidden = int(hidden)
        self._fallback_active = False

    @classmethod
    def get_meta(cls) -> SolverMeta:
        return SolverMeta(
            description="Surrogate-Laplacian 1D diffusion solver, physics fallback",
            kind="hybrid.nn.laplacian",
            equation="du/dt = NN(stencil(u))",
            equation_expr="explicit surrogate Laplacian, dormant FVM fallback",
            dimension=MeshDimension.D1,
            fields={"u": FieldMeta(name="u", vtype=VariableType.SCALAR)},
        )

    @classmethod
    def get_name(cls) -> str:
        return "HybLapSolver"

    @classmethod
    def get_config_class(cls) -> type[SolverConfig]:
        return HybLapSolverConfig

    # -- lifecycle ----------------------------------

    def initialize(self, boundaries=None):
        super().initialize(boundaries)
        # build the dormant physics operator so the fallback swap is hot
        self._physics_op.build(self._mesh, self._backend)

    # -- fallback control ---------------------------

    @property
    def fallback_active(self) -> bool:
        return self._fallback_active

    def fallback_to_physics(self):
        """Swap the surrogate out, dormant FVM physics in. Hot: the current
        field state, ICs and boundary bindings are preserved."""
        if self._fallback_active:
            return
        self._ops = [self._physics_op]
        self._fallback_active = True

    def resume_surrogate(self):
        """Swap the (typically retrained) surrogate back in."""
        if not self._fallback_active:
            return
        self._ops = [self._surrogate_op]
        self._fallback_active = False

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
