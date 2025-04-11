# -*- encoding: utf-8 -*-
"""
Copyright (C) 2025, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Linear algebra class.
"""
from core.numerics.matrix import Matrix
from core.numerics.fields import Field
import numpy as np


class LinearEqs:
    """
    Linear equations solver.
    """

    def __init__(self, variable: str, mat: Matrix, rhs: Matrix):
        """Linear equations solver.

        Args:
            variable: The target variable.
            mat: The coefficient matrix of the linear equations.
            rhs: The right-hand side of the linear equations.
        """
        self._var = variable
        self._mat = mat
        self._rhs = rhs

    # -----------------------------------------------
    # --- static methods ---
    # -----------------------------------------------

    @staticmethod
    def zeros(variable: str, size: int, type: str = "float") -> "LinearEqs":
        """Create a linear equations with all elements set to zero."""
        mat = Matrix.zeros((size, size), type)
        rhs = Matrix.zeros((size,), type)
        return LinearEqs(variable, mat, rhs)

    # -----------------------------------------------
    # --- properties ---
    # -----------------------------------------------

    @property
    def size(self) -> int:
        """The size of the linear equations."""
        return self._mat.shape[0]

    @property
    def variable(self) -> str:
        """The target variable."""
        return self._var

    @property
    def matrix(self) -> Matrix:
        """The cooefficient matrix."""
        return self._mat

    @property
    def rhs(self) -> Matrix:
        """The right-hand side."""
        return self._rhs

    # -----------------------------------------------
    # --- overload methods ---
    # -----------------------------------------------

    def __add__(self, other):
        self._check_variable(other)
        if isinstance(other, LinearEqs):
            return LinearEqs(
                self.variable,
                self._mat + other.matrix,
                self._rhs + other.rhs,
            )
        raise ValueError(f"Invalid LinearEqs operation with {type(other)}.")

    def __sub__(self, other):
        return self.__add__(-other)

    def __radd__(self, other):
        return self.__add__(other)

    def __rsub__(self, other):
        return other + (-self)

    def __mul__(self, other):
        self._check_variable(other)
        return LinearEqs(
            self.variable,
            self._mat * other,
            self._rhs * other,
        )

    def __rmul__(self, other):
        self._check_variable(other)
        return LinearEqs(
            self.variable,
            other * self._mat,
            other * self._rhs,
        )

    def __truediv__(self, other):
        self._check_variable(other)
        return LinearEqs(
            self.variable,
            self._mat / other,
            self._rhs / other,
        )

    def __neg__(self):
        return LinearEqs(self._var, -self._mat, -self._rhs)

    def _check_variable(self, other):
        if isinstance(other, LinearEqs) and other.variable != self.variable:
            raise ValueError(
                f"Cannot add equations with different variables: \
                    {self.variable}, {other.variable}."
            )

    # -----------------------------------------------
    # --- linear equations methods ---
    # -----------------------------------------------

    def scalarize(self) -> list["LinearEqs"]:
        """Scalarize the vector equations."""
        if self.matrix.type == "unknown":
            raise ValueError(f"Cannot scalarize unknown matrix.")

        if self._mat.type == "float":
            return [self]

        mats = self._mat.scalarize()
        rhss = self._rhs.scalarize()
        eqs = []
        for mat, rhs in zip(mats, rhss):
            eqs.append(LinearEqs(self.variable, mat, rhs))

        return eqs

    def solve(self, method: str = "numpy") -> np.ndarray:
        """Solve the linear equations."""
        results = []
        for eq in self.scalarize():
            if method == "numpy":
                try:
                    # If the matrix is all zeros, the solution is the right-hand side.
                    if np.all(eq.matrix.data == 0):
                        results.append(eq.rhs.flatten().data)
                    else:
                        # Solve the linear equations using numpy.
                        results.append(
                            np.linalg.solve(eq.matrix.data, eq.rhs.flatten().data)
                        )
                except:
                    raise RuntimeError("Cannot solve linear equations.")
            else:
                raise ValueError(f"Unsupported algorithm {method}.")
        solution = np.array(results).T
        return solution
