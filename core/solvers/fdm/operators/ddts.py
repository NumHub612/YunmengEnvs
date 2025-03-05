# -*- encoding: utf-8 -*-
"""
Copyright (C) 2025, The YunmengEnvs Contributors. Join us, share your ideas!  

DDt operators for the finite difference method.
"""
from core.solvers.interfaces.IEquation import IOperator
from core.numerics.matrix import LinearEqs, Matrix
from core.numerics.fields import Field, NodeField, Variable, Vector, Scalar, Tensor
from core.numerics.mesh import Mesh


class Ddt01(IOperator):
    """
    Simple first order time derivative operator in fdms.
    """

    def __init__(self):
        self._dt = None

    @property
    def type(self) -> str:
        return "DDT"

    def prepare(self, mesh: Mesh = None, step: float = 0.001, **kwargs):
        self._dt = step

    def run(self, source: Field) -> Variable | LinearEqs:
        # basic information
        size = source.size
        type = source.dtype

        # create matrix
        coef_val = 1.0 / self._dt
        if type == "scalar":
            coef = Scalar.unit() * coef_val
        elif type == "vector":
            coef = Vector.unit() * coef_val
        else:
            coef = Tensor.unit() * coef_val

        mat = Matrix.zeros((size, size), type)
        for i in range(size):
            mat[i, i] = coef

        # create rhs
        rhs = Matrix.zeros((size,), type)
        for i in range(size):
            rhs[i] = source[i] * coef_val

        return LinearEqs(source.variable, mat, rhs)
