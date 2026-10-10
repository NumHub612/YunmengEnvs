# -*- encoding: utf-8 -*-
"""
Copyright (C) 2026, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

description
"""

from yunmeng.interfaces.supports import IMatrix, ILinearEqs, IBackend
from yunmeng.interfaces.types import ArrayLike

from yunmeng.numerics.linalgs.matrix1 import NumpyMatrix
from yunmeng.numerics.linalgs.matrix2 import TorchMatrix


def get_matrix_factory(backend: IBackend) -> IMatrix:
    if backend.name == "numpy":
        return NumpyMatrix
    elif backend.name == "torch":
        return TorchMatrix
    else:
        raise ValueError(f"Unknown backend: {backend}")


def assemble_matrix(equations: list[ILinearEqs], rhs: ArrayLike):
    """Assemble  matrix."""
    matrix, rhs = None, None
    for eqs in equations:
        if matrix is None:
            matrix = eqs.matrix
            rhs = eqs.rhs.values
        else:
            size = matrix.shape[0]
            identity = get_matrix_factory(matrix.btype).identity(
                size, device=matrix.btype.device
            )
            matrix = matrix + eqs.matrix - identity
            rhs = rhs + eqs.rhs.values - rhs
    return matrix, rhs
