# -*- encoding: utf-8 -*-
"""
Copyright (C) 2025, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

The divergence operators for the finite volume method.
"""
from core.solvers.interfaces import IOperator, OperatorType
from core.numerics.mats import LinearEqs
from core.numerics.fields import Field, NodeField, Tensor, Vector, VariableType
from core.numerics.mesh import Grid, ElementType

import numpy as np


class Div01(IOperator):
    """


    scheme:
    """

    @classmethod
    def get_type(self) -> OperatorType:
        return OperatorType.DIV

    def get_name(cls) -> str:
        return "div01"

    def __init__(self):
        self._mesh = None
        self._topo = None
        self._geom = None

    def prepare(self, mesh: Grid, **kwargs):
        if not isinstance(mesh, Grid):
            raise ValueError("Fvm Grad01 operator only supports Grid.")

        self._mesh = mesh
        self._topo = self._mesh.get_topo_assistant()
        self._geom = self._mesh.get_geom_assistant()

    def run(self, source: Field) -> Field | LinearEqs:
        pass
