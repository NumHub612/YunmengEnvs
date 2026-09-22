# -*- encoding: utf-8 -*-
"""
Copyright (C) 2026, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

description
"""

from __future__ import annotations
from dataclasses import asdict, dataclass
from typing import Any, Callable, Sequence
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
from yunmeng.interfaces.solver import SolverConfig, SolverMeta
from yunmeng.solvers.commons import BaseSolver
from yunmeng.operators.FdmDiffusionOp import FdmDiffusionOperator
from yunmeng.operators.NnCorrectionOp import NeuralCorrectionOperator
from yunmeng.numerics.algos import ym_register


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


@ym_register("solver")
class FdmSolver(BaseSolver):
    """Pure-physics backward-Euler FDM diffusion solver."""

    def __init__(
        self,
        sid: str,
        mesh: IMesh,
        config: FdmSolverConfig,
        backend: IBackend,
        nu: float = 0.02,
    ):
        super().__init__(
            sid,
            mesh,
            [FdmDiffusionOperator(config.solution_field, nu)],
            config,
            backend,
        )
        self._nu = float(nu)

    @classmethod
    def get_meta(cls) -> SolverMeta:
        return SolverMeta(
            description="Pure-FDM 1D diffusion solver",
            kind="fdm.diffusion",
            equation="du/dt = nu * d2u/dx2",
            equation_expr="backward-Euler finite differences",
            dimension=MeshDimension.D1,
            fields={"u": FieldMeta(name="u", vtype=VariableType.SCALAR)},
        )

    @classmethod
    def get_name(cls) -> str:
        return "FdmSolver"

    @classmethod
    def get_config_class(cls) -> type[SolverConfig]:
        return FdmSolverConfig
