# -*- encoding: utf-8 -*-
"""
Copyright (C) 2025, The YunmengEnvs Contributors. Join us, share your ideas!  

DDt operators for the finite difference method.
"""
from core.solvers.interfaces.IEquation import IOperator
from core.numerics.matrix import LinearEqs, Matrix
from core.numerics.fields import Field, Variable
from core.numerics.mesh import MeshGeom, MeshTopo


class Ddt01(IOperator):
    """
    Simple first order time derivative operator in fdms.
    """

    def __init__(self):
        self._field = None
        self._dt = None

    @property
    def type(self) -> str:
        return "DDT"

    def prepare(self, field: Field, topo: MeshTopo, geom: MeshGeom, **kwargs):
        self._field = field
        self._dt = kwargs.get("dt", 0.001)

    def run(self, element: int) -> Variable | LinearEqs:
        # basic information
        var = self._field.variable
        size = self._field.size
        type = self._field.dtype

        # create matrix
        coef = 1.0 / self._dt
        mat = Matrix.zeros((size, size), type)
        mat[element, element] = coef

        # create rhs
        src = self._field.at(element) * coef
        rhs = Matrix.zeros((size, 1), type)
        rhs[element, 0] = src

        return LinearEqs(var, mat, rhs)
