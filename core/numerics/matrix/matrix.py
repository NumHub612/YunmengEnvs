# -*- encoding: utf-8 -*-
"""
Copyright (C) 2025, The YunmengEnvs Contributors. Join us, share your ideas!  

Matrix and linear equations.
"""
from core.numerics.fields import Variable, Scalar, Vector, Tensor
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
        self._data = np.array(data)
        if self.type == "unknown":
            raise ValueError(f"Unsupported matrix type.")

        if shape is not None:
            self._data = self._data.reshape(shape)

    # -----------------------------------------------
    # --- static methods ---
    # -----------------------------------------------

    @staticmethod
    def zeros(shape: tuple, type: str = "float") -> "Matrix":
        """Create a matrix with all elements set to zero."""
        if type == "float":
            return Matrix(np.zeros(shape), shape)

        if type == "scalar":
            zero = Scalar().zero()
        elif type == "vector":
            zero = Vector().zero()
        elif type == "tensor":
            zero = Tensor().zero()
        else:
            raise ValueError(f"Invalid matrix type {type}.")

        return Matrix(np.full(shape, zero), shape)

    @staticmethod
    def ones(shape: tuple, type: str = "float") -> "Matrix":
        """Create a matrix with all elements set to one."""
        if type == "float":
            return Matrix(np.ones(shape), shape)

        if type == "scalar":
            one = Scalar().unit()
        elif type == "vector":
            one = Vector().unit()
        elif type == "tensor":
            one = Tensor().unit()
        else:
            raise ValueError(f"Invalid matrix type {type}.")

        return Matrix(np.full(shape, one), shape)

    @staticmethod
    def unit(shape: tuple, type: str = "float") -> "Matrix":
        """Create a Identity Matrix."""
        if len(shape) == 2 and shape[0] != shape[1]:
            raise ValueError("Unit matrix must be squared.")

        if type == "float":
            return Matrix(np.identity(shape[0]), shape)

        if type == "scalar":
            one = Scalar().unit()
        elif type == "vector":
            one = Vector().unit()
        elif type == "tensor":
            one = Tensor().unit()
        else:
            raise ValueError(f"Invalid matrix type {type}.")

        return Matrix(np.identity(shape[0]) * one, shape)

    # -----------------------------------------------
    # --- properties ---
    # -----------------------------------------------

    @property
    def shape(self) -> tuple:
        return self._data.shape

    @property
    def data(self) -> np.ndarray:
        return self._data

    @property
    def diag(self) -> np.ndarray:
        return np.diag(self._data)

    @property
    def type(self) -> str:
        """The matrix data type, e.g. float, scalar, vector, tensor, etc."""
        # TODO: check all elements are of the same type.

        if len(self.shape) == 2 and self.shape[1] > 0:
            elem = self._data[0][0]
        else:
            elem = self._data[0]

        if isinstance(elem, (int, float)):
            return "float"
        elif isinstance(elem, Variable):
            return elem.type
        else:
            return "unknown"

    # -----------------------------------------------
    # --- overload methods ---
    # -----------------------------------------------

    def __getitem__(self, index: tuple):
        return self._data[index]

    def __setitem__(self, index: tuple, value: Variable):
        self._data[index] = value

    def __add__(self, other):
        if isinstance(other, Variable):
            return Matrix(self._data + other)
        return Matrix(self._data + other.data)

    def __sub__(self, other):
        if isinstance(other, Variable):
            return Matrix(self._data - other)
        return Matrix(self._data - other.data)

    def __mul__(self, other):
        if isinstance(other, Variable):
            return Matrix(self._data * other)
        return Matrix(self._data * other.data)

    def __truediv__(self, other):
        return Matrix(self._data / other)

    def __rtruediv__(self, other):
        return Matrix(other / self._data)

    def __rmul__(self, other):
        return Matrix(other * self._data)

    def __radd__(self, other):
        return Matrix(other + self._data)

    def __rsub__(self, other):
        return Matrix(other - self._data)

    # -----------------------------------------------
    # --- matrix methods ---
    # -----------------------------------------------

    def transpose(self) -> "Matrix":
        return Matrix(self._data.transpose())

    def inverse(self) -> "Matrix":
        return Matrix(np.linalg.inv(self._data))

    def flatten(self) -> "Matrix":
        return Matrix(self._data.flatten())

    def abs(self) -> "Matrix":
        return Matrix(np.abs(self._data))

    # -----------------------------------------------
    # --- extended methods ---
    # -----------------------------------------------

    def scalarize(self) -> list["Matrix"]:
        """Scalarize the vector equations."""
        if self.type == "unknown":
            raise ValueError(f"Cannot scalarize unknown matrix.")

        if self.type == "float":
            return [self]

        type_dims = {"scalar": 1, "vector": 3, "tensor": 9}
        dim = type_dims[self.type]

        mats = [self.zeros(self.shape) for _ in range(dim)]
        if len(self.shape) == 1:
            for i in range(self.shape[0]):
                arr = self._data[i].to_np().flatten()
                for k in range(dim):
                    mats[k][(i,)] = arr[k]
        else:
            for i in range(self.shape[0]):
                for j in range(self.shape[1]):
                    arr = self._data[(i, j)].to_np().flatten()
                    for k in range(dim):
                        mats[k][(i, j)] = arr[k]

        return mats


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
        if isinstance(other, (Variable, int, float)):
            return LinearEqs(
                self.variable,
                self._mat + other,
                self._rhs + other,
            )
        if isinstance(other, LinearEqs):
            return LinearEqs(
                self.variable,
                self._mat + other.matrix,
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
        if isinstance(other, LinearEqs) and other.variable != self.variable:
            raise ValueError(
                f"Cannot add equations with different variables: \
                    {self.variable}, {other.variable}."
            )
        if isinstance(other, Variable):
            if other.type != self.matrix.type:
                raise ValueError(
                    f"Cannot add variable with different type: \
                        {self.matrix.type}, {other.type}."
                )

    def _valid_multiple(self, other):
        if not isinstance(other, (Variable, int, float)):
            raise ValueError(f"Cannot multiply equations with {type(other)}.")

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
                    # If the matrix is singular, the solution is all zeros.
                    results.append(np.zeros(eq.size))
            else:
                raise ValueError(f"Unsupported algorithm {method}.")
        solution = np.array(results).T
        return solution
