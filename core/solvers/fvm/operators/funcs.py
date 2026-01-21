# -*- encoding: utf-8 -*-
"""
Copyright (C) 2025, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

The function operators used in FVM.
"""
from core.solvers.interfaces import (
    IBoundaryCondition,
    IOperator,
    OperatorType,
    BoundaryType,
)
from core.numerics.mats import LinearEqs
from core.numerics.fields import Field, Scalar, DataHub
from core.numerics.mesh import Grid


class CFL01(IOperator):
    """
    The CFL01 operator used for calculating the CFL number.
    """

    @classmethod
    def get_name(cls) -> str:
        return "CFL01"

    @classmethod
    def get_type(cls) -> OperatorType:
        return OperatorType.FUNC

    def prepare(self, vars: list[str], mesh: Grid, boundaries: dict):
        pass

    def run(self, source: DataHub) -> Scalar:
        pass
