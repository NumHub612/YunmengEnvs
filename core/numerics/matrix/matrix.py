# -*- encoding: utf-8 -*-
"""
Copyright (C) 2025, The YunmengEnvs Contributors. Join us, share your ideas!  

Matrix and linear equations.
"""
from core.numerics.fields import Variable
import numpy as np


class Matrix:
    """
    Matrix class.
    """

    def __init__(self, data: np.ndarray | list, shape: tuple = None):
        """Matrix class.

        Args:
            data: The data of the matrix.
            shape: The matrix shape.
        """
        self.data = np.array(data)
        if shape is not None:
            self.data = self.data.reshape(shape)

    @staticmethod
    def zeros(shape: tuple) -> "Matrix":
        """Create a matrix with all elements set to zero."""
        return Matrix(np.zeros(shape), shape)

    @staticmethod
    def ones(shape: tuple) -> "Matrix":
        """Create a matrix with all elements set to one."""
        return Matrix(np.ones(shape), shape)

    @staticmethod
    def unit(shape: tuple) -> "Matrix":
        """Create a Identity Matrix."""
        return Matrix(np.identity(shape[0]), shape)

    @property
    def shape(self) -> tuple:
        return self.data.shape

    @property
    def raw(self) -> np.ndarray:
        return self.data

    @property
    def rank(self) -> int:
        return np.linalg.matrix_rank(self.data)

    @property
    def magnitude(self) -> float:
        return np.linalg.norm(self.data)

    @property
    def trace(self) -> float:
        return np.trace(self.data)

    @property
    def determinant(self) -> float:
        return np.linalg.det(self.data)

    @property
    def dtype(self) -> str:
        elem = self.data[0] if len(self.shape) == 1 else self.data[0][0]

        if isinstance(elem, Variable):
            return elem.type
        elif isinstance(elem, (int, float)):
            return "float"
        else:
            return "unknown"

    def __getitem__(self, index: tuple):
        return self.data[index]

    def __setitem__(self, index: tuple, value: Variable):
        self.data[index] = value

    def __add__(self, other):
        if isinstance(other, Variable):
            return Matrix(self.data + other)
        return Matrix(self.data + other.data)

    def __sub__(self, other):
        if isinstance(other, Variable):
            return Matrix(self.data - other)
        return Matrix(self.data - other.data)

    def __mul__(self, other):
        if isinstance(other, Variable):
            return Matrix(self.data * other)
        return Matrix(self.data * other.data)

    def __truediv__(self, other):
        return Matrix(self.data / other)

    def __rtruediv__(self, other):
        return Matrix(other / self.data)

    def __rmul__(self, other):
        return Matrix(other * self.data)

    def __radd__(self, other):
        return Matrix(other + self.data)

    def __rsub__(self, other):
        return Matrix(other - self.data)

    def transpose(self) -> "Matrix":
        return Matrix(self.data.transpose())

    def dot(self, other: "Matrix") -> "Matrix":
        return Matrix(np.dot(self.data, other.data))

    def inverse(self) -> "Matrix":
        return Matrix(np.linalg.inv(self.data))

    def flatten(self) -> "Matrix":
        return Matrix(self.data.flatten())

    def abs(self) -> "Matrix":
        return Matrix(np.abs(self.data))


class LinearEqs:
    """
    Linear equations solver.
    """

    def __init__(self, variable: str, mat: Matrix, rhs: Matrix):
        """Linear equations solver.

        Args:
            variable: The target variable.
            mat: The cooefficient matrix of the linear equations.
            rhs: The right-hand side of the linear equations.
            algo: The algorithm to solve.
        """
        self._var = variable
        self._mat = mat
        self._rhs = rhs

    @staticmethod
    def zeros(variable: str, size: int) -> "LinearEqs":
        """Create a linear equations with all elements set to zero."""
        mat = Matrix.zeros((size, size))
        rhs = Matrix.zeros((size, 1))
        return LinearEqs(variable, mat, rhs)

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

    def __add__(self, other):
        self._check_variable(other)
        if isinstance(other, (Variable, int, float)):
            return LinearEqs(
                self.variable,
                self._mat + other,
                self._rhs + other,
            )
        if isinstance(other, LinearEqs):
            return LinearEqs(
                self.variable,
                self._mat + other.mat,
                self._rhs + other.rhs,
            )
        raise ValueError(f"Invalid LinearEqs operation.")

    def __sub__(self, other):
        return self.__add__(-other)

    def __radd__(self, other):
        return self.__add__(other)

    def __rsub__(self, other):
        return other + (-self)

    def __mul__(self, other):
        self._check_variable(other)
        self._valid_multiple(other)
        return LinearEqs(
            self.variable,
            self._mat * other,
            self._rhs * other,
        )

    def __rmul__(self, other):
        self._check_variable(other)
        self._valid_multiple(other)
        return LinearEqs(
            self.variable,
            other * self._mat,
            other * self._rhs,
        )

    def __truediv__(self, other):
        self._check_variable(other)
        self._valid_multiple(other)
        return LinearEqs(
            self.variable,
            self._mat / other,
            self._rhs / other,
        )

    def _check_variable(self, other):
        if other.variable != self.variable:
            raise ValueError(
                f"Cannot add equations with different variables: \
                    {self.variable}, {other.variable}."
            )

    def _valid_multiple(self, other):
        if not isinstance(other, (Variable, int, float)):
            raise ValueError(f"Cannot multiply equations with {type(other)}.")

    def scalarize(self) -> list["LinearEqs"]:
        """Scalarize the vector equations."""
        if self._mat.dtype == "float":
            return [self]

        if self._mat.dtype == "scalar":
            pass
        elif self._mat.dtype == "vector":
            pass
        else:
            raise ValueError(f"Unsupported matrix dtype {self._mat.dtype}.")

    def solve(self, method: str = "numpy") -> np.ndarray:
        """Solve the linear equations."""
        results = []
        for eq in self.scalarize():
            if method == "numpy":
                results.append(np.linalg.solve(eq.matrix.data, eq.rhs.data))
            else:
                raise ValueError(f"Unsupported algorithm {method}.")

        return np.array(results)
