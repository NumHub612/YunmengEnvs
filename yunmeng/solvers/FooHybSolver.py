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


class HybridSolver(BaseSolver):
    """Physics+neural hybrid diffusion solver."""

    def __init__(
        self,
        sid: str,
        mesh: IMesh,
        config: HybridSolverConfig,
        backend: IBackend,
        nu: float = 0.02,
        hidden: int = 32,
        operators: Sequence = None,
    ):
        if operators is None:
            operators = [
                FdmDiffusionOperator(config.solution_field, nu),
                NeuralCorrectionOperator(
                    config.solution_field,
                    hidden=hidden,
                    device=config.device,
                ),
            ]
        super().__init__(sid, mesh, operators, config, backend)
        self._nu = float(nu)
        self._hidden = int(hidden)

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
