# -*- encoding: utf-8 -*-
"""
Copyright (C) 2025, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

DDt operators for the finite difference method.
"""
from core.solvers.interfaces.IEquation import IOperator
from core.numerics.matrix import LinearEqs, DenseMatrix
from core.numerics.fields import Field, NodeField, Variable, Vector, Tensor
from core.numerics.mesh import Mesh


class Ddt01(IOperator):
    """
    Simple first order time derivative operator in fdms.
    """

    @property
    def type(self) -> str:
        return "DDT"

    def prepare(self, mesh: Mesh = None, step: float = 0.001, **kwargs):
        self._dt = step

    def run(self, source: Field) -> Variable | LinearEqs:
        # basic information
        variable = source.variable
        size = source.size
        dtype = source.dtype

        # create matrix
        coef_val = 1.0 / self._dt
        if dtype == "vector":
            coef = Vector.unit() * coef_val
        elif dtype == "tensor":
            coef = Tensor.unit() * coef_val

        mat = DenseMatrix.zeros((size, size), dtype)
        for i in range(size):
            mat[i, i] = coef

        # create rhs
        rhs = NodeField(size, dtype)
        for i in range(size):
            rhs[i] = source[i] * coef_val

        return LinearEqs(variable, mat, rhs)
