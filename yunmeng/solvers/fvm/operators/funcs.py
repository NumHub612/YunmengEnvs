# -*- encoding: utf-8 -*-
"""
Copyright (C) 2025, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

The function operators used in FVM.
"""

from yunmeng.solvers.interfaces import (
    IBoundaryCondition,
    IOperator,
    OperatorType,
    BoundaryType,
)
from yunmeng.numerics.grids import Grid
from yunmeng.numerics.fields import Field, Variable
from yunmeng.numerics.fields.datahubs import DataHub


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

    def prepare(self, fields: list[str], mesh: Grid, bounds: dict):
        pass

    def forward(self, sources: DataHub) -> Variable:
        pass
