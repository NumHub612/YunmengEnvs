# -*- encoding: utf-8 -*-
"""
Copyright (C) 2025, The YunmengEnvs Contributors. Join us, share your ideas!  

Matrix.
"""
from core.numerics.fields import Variable
import numpy as np


class Matrix:
    """
    Matrix class.
    """

    def __init__(self, data: np.ndarray | list, shape: tuple):
        """Matrix class.

        Args:
            data: The data of the matrix.
            shape: The matrix shape.
        """
        if len(shape) > 2:
            raise ValueError(f"Invalid shape {shape}.")

        self.data = np.array(data).reshape(shape)

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
    def data(self) -> np.ndarray:
        return self.data

    @property
    def rank(self) -> int:
        return np.linalg.matrix_rank(self.data)

    @property
    def magnitude(self) -> float:
        return np.linalg.norm(self.data)

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

    def __add__(self, other: "Matrix" | Variable):
        if isinstance(other, Variable):
            return Matrix(self.data + other)
        return Matrix(self.data + other.data)

    def __sub__(self, other: "Matrix" | Variable):
        if isinstance(other, Variable):
            return Matrix(self.data - other)
        return Matrix(self.data - other.data)

    def __mul__(self, other: "Matrix" | Variable):
        if isinstance(other, Variable):
            return Matrix(self.data * other)
        return Matrix(self.data * other.data)

    def __truediv__(self, other: Variable):
        return Matrix(self.data / other)

    def __rtruediv__(self, other: Variable):
        return Matrix(other / self.data)

    def __rmul__(self, other: Variable):
        return Matrix(other * self.data)

    def __radd__(self, other: Variable):
        return Matrix(other + self.data)

    def __rsub__(self, other: Variable):
        return Matrix(other - self.data)

    def transpose(self):
        return Matrix(self.data.transpose())

    def dot(self, other: "Matrix"):
        return Matrix(np.dot(self.data, other.data))

    def trace(self):
        return np.trace(self.data)

    def determinant(self):
        return np.linalg.det(self.data)

    def inverse(self):
        return Matrix(np.linalg.inv(self.data))

    def flatten(self):
        return Matrix(self.data.flatten())

    def abs(self):
        return Matrix(np.abs(self.data))


class LinearEqs:
    """
    Linear equations solver.
    """

    def __init__(self, variable: str, mat: Matrix, rhs: Matrix, algo: str = "numpy"):
        """Linear equations solver.

        Args:
            variable: The target variable.
            mat: The cooefficient matrix of the linear equations.
            rhs: The right-hand side of the linear equations.
            algo: The algorithm to solve the linear equations.
        """
        self.variable = variable
        self.algo = algo
        self.mat = mat
        self.rhs = rhs

    @property
    def size(self) -> int:
        """The size of the linear equations."""
        return self.mat.shape[0]

    @property
    def variable(self) -> str:
        """The target variable."""
        return self.variable

    @property
    def matrix(self) -> Matrix:
        """The cooefficient matrix."""
        return self.mat

    @property
    def rhs(self) -> Matrix:
        """The right-hand side."""
        return self.rhs

    def __add__(self, other: "LinearEqs"):
        return LinearEqs(self.mat + other.mat, self.rhs + other.rhs, self.algo)

    def __sub__(self, other: "LinearEqs"):
        return LinearEqs(self.mat - other.mat, self.rhs - other.rhs, self.algo)

    def scalarize(self) -> list["LinearEqs"]:
        """Scalarize the vector equations."""
        if self.mat.dtype == "float":
            return [self]

        if self.mat.dtype == "scalar":
            pass
        elif self.mat.dtype == "vector":
            pass
        else:
            raise ValueError(f"Unsupported matrix dtype {self.mat.dtype}.")

    def solve(self) -> np.ndarray:
        """Solve the linear equations."""
        results = []
        for eq in self.scalarize():
            if eq.algo == "numpy":
                results.append(np.linalg.solve(eq.mat.data, eq.rhs.data))
            else:
                raise ValueError(f"Unsupported algorithm {eq.algo}.")

        return np.array(results)
