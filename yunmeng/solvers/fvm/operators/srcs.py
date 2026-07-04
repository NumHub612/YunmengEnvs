# -*- encoding: utf-8 -*-
"""
Copyright (C) 2025, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Solution of the 2D source term equation using finite volume method.
"""

from yunmeng.solvers.interfaces import IOperator, OperatorType
from yunmeng.numerics.linalgs import LinearEqs
from yunmeng.numerics.grids import Grid
from yunmeng.numerics.fields import Field, DataHub


import numpy as np


class Src01(IOperator):
    """
    Simple implicit source term operator excuted on `Grid` mesh in fvm.

    scheme:
        - implicit method.
    """

    @classmethod
    def get_type(cls) -> OperatorType:
        return OperatorType.SRC

    @classmethod
    def get_name(cls) -> str:
        return "src01"

    def __init__(self):
        self._mesh = None
        self._topo = None
        self._geom = None
        self._part = None
        self._var = ""

    def prepare(self, fields: list[str], mesh: Grid, bounds: dict):
        if not isinstance(mesh, Grid):
            raise ValueError("Fvm Src01 operator only supports Grid.")

        self._mesh = mesh
        self._topo = self._mesh.get_topo_assistant()
        self._geom = self._mesh.get_geom_assistant()
        self._part = self._mesh.get_part_assistant()
        self._var = fields[0]

    def forward(self, sources: DataHub) -> Field | LinearEqs:
        source = sources.field(self._var).data
        src_eqs = LinearEqs.zeros(self._part, rhs_type=source.dtype, etype=source.etype)
        return src_eqs
