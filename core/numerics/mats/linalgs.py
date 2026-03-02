# -*- encoding: utf-8 -*-
"""
Copyright (C) 2025, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Linear algebra class.
"""
from core.numerics.enums import VariableType, ElementType
from core.numerics.mats import Matrix, SparseMatrix
from core.numerics.algos import MeshPart
from core.numerics.fields import Field

from typing import Callable
import numpy as np
import torch


class LinearEqs:
    """
    Linear equations solver.
    """

    def __init__(self, mat: Matrix, rhs: Field):
        self._mat = mat
        self._rhs = rhs

        if mat.shape[0] != mat.shape[1] or mat.shape[0] != rhs.size:
            raise ValueError(
                f"The matrix is not square: {mat.shape}, or not compatible \
                    with the rhs: {rhs.size}."
            )
        if mat.dtype != VariableType.SCALAR and mat.dtype != rhs.dtype:
            raise ValueError(
                f"The matrix type {mat.dtype} is not compatible with the \
                    rhs type {rhs.dtype}."
            )

        self._size = self._rhs.size

    # -----------------------------------------------
    # region static methods
    # -----------------------------------------------

    @staticmethod
    def zeros(
        partitions: MeshPart,
        matrix_type: VariableType = VariableType.SCALAR,
        rhs_type: VariableType = VariableType.SCALAR,
        etype: ElementType = ElementType.CELL,
    ) -> "LinearEqs":
        """Create a linear equations with all elements set to zero."""
        size = partitions.get_size(etype)
        mat = SparseMatrix.zeros((size, size), matrix_type)
        rhs = Field(partitions, rhs_type, etype)
        return LinearEqs(mat, rhs)

    # -----------------------------------------------
    # region properties
    # -----------------------------------------------

    @property
    def size(self) -> int:
        """The size of the linear equations."""
        return self._size

    @property
    def matrix(self) -> Matrix:
        """The cooefficient matrix."""
        return self._mat

    @matrix.setter
    def matrix(self, value: Matrix):
        """Set the coefficient matrix."""
        if not isinstance(value, Matrix):
            raise ValueError(f"Invalid matrix type: {type(value)}.")
        if self._mat.shape != value.shape:
            raise ValueError(f"Invalid matrix shape: {value.shape}.")
        if self._mat.dtype != value.dtype:
            raise ValueError(f"Invalid matrix type: {value.dtype}.")
        self._mat = value

    @property
    def rhs(self) -> Field:
        """The right-hand side vector."""
        return self._rhs

    @rhs.setter
    def rhs(self, value: Field):
        """Set the right-hand side."""
        if not isinstance(value, Field):
            raise ValueError(f"Invalid rhs type: {type(value)}.")
        if self._rhs.size != value.size:
            raise ValueError(f"Invalid rhs size: {value.size}.")
        if self._rhs.dtype != value.dtype:
            raise ValueError(f"Invalid rhs type: {value.dtype}.")
        self._rhs = value

    # -----------------------------------------------
    # region operations methods
    # -----------------------------------------------

    def __add__(self, other: "LinearEqs"):
        self._check_compatible(other)
        return LinearEqs(
            self._mat + other.matrix,
            self._rhs + other.rhs,
        )

    def __radd__(self, other):
        return self.__add__(other)

    def __iadd__(self, other: "LinearEqs"):
        self._check_compatible(other)
        self._mat += other.matrix
        self._rhs += other.rhs
        return self

    def __sub__(self, other: "LinearEqs"):
        self._check_compatible(other)
        return LinearEqs(
            self._mat - other.matrix,
            self._rhs - other.rhs,
        )

    def __rsub__(self, other: "LinearEqs"):
        self._check_compatible(other)
        return LinearEqs(
            other.matrix - self._mat,
            other.rhs - self._rhs,
        )

    def __isub__(self, other: "LinearEqs"):
        self._check_compatible(other)
        self._mat -= other.matrix
        self._rhs -= other.rhs
        return self

    def __neg__(self):
        return LinearEqs(-self._mat, -self._rhs)

    def _check_compatible(self, other):
        if not isinstance(other, LinearEqs):
            raise ValueError(f"Invalid LinearEqs operation with {type(other)}.")
        if other.size != self.size:
            raise ValueError(
                f"Invalid LinearEqs operation with different sizes: \
                    {self.size} vs {other.size}."
            )
        if other.matrix.dtype != self.matrix.dtype:
            raise ValueError(
                f"Invalid LinearEqs operation with different matrix types: \
                    {self.matrix.dtype} vs {other.matrix.dtype}."
            )
        if other.rhs.dtype != self.rhs.dtype:
            raise ValueError(
                f"Invalid LinearEqs operation with different rhs types: \
                    {self.rhs.dtype} vs {other.rhs.dtype}."
            )

    # -----------------------------------------------
    # region solve methods
    # -----------------------------------------------

    def scalarize(self) -> list["LinearEqs"]:
        """Scalarize the vector equations."""
        mat_lst = self._mat.scalarize()
        rhs_lst = self._rhs.scalarize()

        eqs = []
        if len(mat_lst) == 1:
            for i, rhs in enumerate(rhs_lst):
                eqs.append(LinearEqs(mat_lst[0], rhs))
        else:
            for mat, rhs, i in zip(mat_lst, rhs_lst, range(len(mat_lst))):
                eqs.append(LinearEqs(mat, rhs))
        return eqs

    def solve(self, engine: Callable) -> Field:
        """Solve the linear equations."""
        results = []
        for eq in self.scalarize():
            result = engine(eq.matrix, eq.rhs)
            results.append(result)

        results = np.array(results).T
        result = Field.from_array(results, self.rhs._mesh_part, self.rhs._meta)
        return result
