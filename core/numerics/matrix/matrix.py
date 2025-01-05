# -*- encoding: utf-8 -*-
"""
Copyright (C) 2025, The YunmengEnvs Contributors. Join us, share your ideas!  

Matrix.
"""
import numpy as np


class Matrix:
    """
    Matrix class.
    """

    def __init__(self, data: np.ndarray | list, shape: tuple):
        self.data = np.array(data).reshape(shape)

    def __add__(self, other: "Matrix" | float):
        if isinstance(other, float):
            return Matrix(self.data + other)
        return Matrix(self.data + other.data)

    def __sub__(self, other: "Matrix" | float):
        if isinstance(other, float):
            return Matrix(self.data - other)
        return Matrix(self.data - other.data)

    def __mul__(self, other: "Matrix" | float):
        if isinstance(other, float):
            return Matrix(self.data * other)
        return Matrix(self.data * other.data)

    def __truediv__(self, other: float):
        return Matrix(self.data / other)

    def __rtruediv__(self, other: float):
        return Matrix(other / self.data)

    def __rmul__(self, other: float):
        return Matrix(other * self.data)

    def __radd__(self, other: float):
        return Matrix(other + self.data)

    def __rsub__(self, other: float):
        return Matrix(other - self.data)

    def __getitem__(self, index: tuple):
        return self.data[index]

    def __setitem__(self, index: tuple, value: float):
        self.data[index] = value

    def transpose(self):
        return Matrix(self.data.transpose())

    def dot(self, other):
        return Matrix(np.dot(self.data, other.data))

    def trace(self):
        return np.trace(self.data)

    def determinant(self):
        return np.linalg.det(self.data)

    def inverse(self):
        return Matrix(np.linalg.inv(self.data))

    def argmax(self, axis=None):
        return Matrix(np.argmax(self.data, axis=axis))

    def argmin(self, axis=None):
        return Matrix(np.argmin(self.data, axis=axis))

    def max(self, axis=None):
        return Matrix(np.max(self.data, axis=axis))

    def min(self, axis=None):
        return Matrix(np.min(self.data, axis=axis))

    def flatten(self):
        return Matrix(self.data.flatten())

    def sum(self, axis=None):
        return Matrix(np.sum(self.data, axis=axis))

    def mean(self, axis=None):
        return Matrix(np.mean(self.data, axis=axis))

    def abs(self):
        return Matrix(np.abs(self.data))


class LinearEqs:
    """
    Linear equations solver.
    """

    def __init__(self, mat: Matrix, rhs: Matrix, algo: str = "numpy"):
        """Linear equations solver.

        Args:
            mat: The cooefficient matrix of the linear equations.
            rhs: The right-hand side of the linear equations.
            algo: The algorithm to solve the linear equations.
        """
        self.mat = mat
        self.rhs = rhs
        self.algo = algo

    @property
    def size(self) -> int:
        """The size of the linear equations."""
        return self.mat.shape[0]

    @property
    def rank(self) -> int:
        """The rank of the matrix."""
        return np.linalg.matrix_rank(self.mat.data)

    @property
    def matrix(self) -> Matrix:
        """The cooefficient matrix of the linear equations."""
        return self.mat

    @property
    def rhs(self) -> Matrix:
        """The right-hand side of the linear equations."""
        return self.rhs

    def solve(self) -> np.ndarray:
        if self.algo == "numpy":
            return np.linalg.solve(self.mat.data, self.rhs.data)
        else:
            raise ValueError("Invalid algorithm.")
