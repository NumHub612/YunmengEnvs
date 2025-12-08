# -*- encoding: utf-8 -*-
"""
Copyright (C) 2025, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Solution of the 2D source term equation using finite volume method.
"""
from core.solvers.interfaces import IOperator, OperatorType
from core.numerics.mats import LinearEqs
from core.numerics.fields import Field, NodeField, Tensor, Vector, DataHub
from core.numerics.mesh import Grid, ElementType

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

    def prepare(self, mesh: Grid, boundaries: dict):
        if not isinstance(mesh, Grid):
            raise ValueError("Fvm Src01 operator only supports Grid.")

        self._mesh = mesh
        self._topo = self._mesh.get_topo_assistant()
        self._geom = self._mesh.get_geom_assistant()

    def run(self, source: DataHub) -> Field | LinearEqs:
        source = source.fetch().data
        variable = source.variable
        src_eqs = LinearEqs.zeros(
            self._mesh.cell_count, rhs_type=source.dtype, variable=variable
        )
        return src_eqs
