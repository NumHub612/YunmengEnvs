# -*- encoding: utf-8 -*-
"""
Copyright (C) 2025, The YunmengEnvs Contributors. Join us, share your ideas!  

DDt operators for the finite difference method.
"""
from core.solvers.interfaces.IEquation import IOperator
from core.numerics.matrix import LinearEqs, Matrix
from core.numerics.fields import Field, Variable, Vector, Scalar, Tensor
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
        coef_val = 1.0 / self._dt
        if type == "scalar":
            coef = Scalar.unit() * coef_val
        elif type == "vector":
            coef = Vector.unit() * coef_val
        else:
            coef = Tensor.unit() * coef_val

        mat = Matrix.zeros((size, size), type)
        mat[element, element] = coef

        # create rhs
        src = self._field[element] * coef_val
        rhs = Matrix.zeros((size,), type)
        rhs[element] = src

        return LinearEqs(var, mat, rhs)
