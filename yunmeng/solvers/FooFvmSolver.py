# -*- encoding: utf-8 -*-
"""
Copyright (C) 2026, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Pure-physics FVM diffusion solver — finite-volume counterpart of FdmSolver.
"""

from __future__ import annotations

from dataclasses import dataclass

from yunmeng.interfaces.solver import SolverConfig, SolverMeta
from yunmeng.interfaces.supports import FieldMeta, IBackend, IMesh
from yunmeng.interfaces.types import MeshDimension, VariableType
from yunmeng.solvers.commons import BaseSolver
from yunmeng.operators.FvmDiffusionOp import FvmDiffusionOperator


@dataclass
class FvmSolverConfig(SolverConfig):
    """Configuration for the pure-FVM diffusion solver."""

    dt: float = 0.01
    t0: float = 0.0
    end_time: float = 1.0
    solution_field: str = "u"
    device: str = "cpu"

    @classmethod
    def get_solver_name(cls) -> str:
        return "FvmSolver"


class FvmSolver(BaseSolver):
    """Pure-physics backward-Euler FVM diffusion solver."""

    def __init__(
        self,
        sid: str,
        mesh: IMesh,
        config: FvmSolverConfig,
        backend: IBackend,
        nu: float = 0.02,
    ):
        super().__init__(
            sid,
            mesh,
            [FvmDiffusionOperator(config.solution_field, nu)],
            config,
            backend,
        )
        self._nu = float(nu)

    @classmethod
    def get_meta(cls) -> SolverMeta:
        return SolverMeta(
            description="Pure-FVM 1D diffusion solver",
            kind="fvm.diffusion",
            equation="d/dt ∫u dV = ∮ nu * grad(u) · n dA",
            equation_expr="backward-Euler finite volumes",
            dimension=MeshDimension.D1,
            fields={"u": FieldMeta(name="u", vtype=VariableType.SCALAR)},
        )

    @classmethod
    def get_name(cls) -> str:
        return "FvmSolver"

    @classmethod
    def get_config_class(cls) -> type[SolverConfig]:
        return FvmSolverConfig
