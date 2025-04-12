# -*- encoding: utf-8 -*-
"""
Copyright (C) 2025, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Linear algebra class.
"""
from core.numerics.matrix import Matrix, DenseMatrix
from core.numerics.fields import Field
import numpy as np
import torch


class LinearEqs:
    """
    Linear equations solver.
    """

    def __init__(self, variable: str, mat: Matrix, rhs: Field):
        """Linear equations solver.

        Args:
            variable: The target variable.
            mat: The coefficient matrix of the linear equations.
            rhs: The right-hand side of the linear equations.
        """
        self._var = variable
        self._mat = mat
        self._rhs = rhs

        if self._mat.shape[0] != self._rhs.size:
            raise ValueError(
                f"The size of the matrix ({self._mat.shape[0]}) is not equal to \
                    the size of the right-hand side ({self._rhs.size})."
            )
        self._size = self._rhs.size

    # -----------------------------------------------
    # --- static methods ---
    # -----------------------------------------------

    @staticmethod
    def zeros(
        variable: str, size: int, matrix_type: str = "float", rhs_type: str = "float"
    ) -> "LinearEqs":
        """Create a linear equations with all elements set to zero."""
        mat = DenseMatrix.zeros((size, size), matrix_type)
        rhs = Field(size, "none", rhs_type)
        return LinearEqs(variable, mat, rhs)

    # -----------------------------------------------
    # --- properties ---
    # -----------------------------------------------

    @property
    def size(self) -> int:
        """The size of the linear equations."""
        return self._size

    @property
    def variable(self) -> str:
        """The target variable."""
        return self._var

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
        if self._mat.type != value.type:
            raise ValueError(f"Invalid matrix type: {value.type}.")
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
    # --- overload methods ---
    # -----------------------------------------------

    def __add__(self, other: "LinearEqs"):
        self._check_compatible(other)
        return LinearEqs(
            self.variable,
            self._mat + other.matrix,
            self._rhs + other.rhs,
        )

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        return self.__add__(-other)

    def __rsub__(self, other):
        return other + (-self)

    def __neg__(self):
        return LinearEqs(self._var, -self._mat, -self._rhs)

    def _check_compatible(self, other):
        if not isinstance(other, LinearEqs):
            raise ValueError(f"Invalid LinearEqs operation with {type(other)}.")
        if other.size != self.size:
            raise ValueError(
                f"Invalid LinearEqs operation with different sizes: \
                    {self.size} vs {other.size}."
            )
        if other.variable != self.variable:
            raise ValueError(
                f"Invalid LinearEqs operation with different variables: \
                    {self.variable} vs {other.variable}."
            )
        if other.matrix.type != self.matrix.type:
            raise ValueError(
                f"Invalid LinearEqs operation with different matrix types: \
                    {self.matrix.type} vs {other.matrix.type}."
            )
        if other.rhs.dtype != self.rhs.dtype:
            raise ValueError(
                f"Invalid LinearEqs operation with different rhs types: \
                    {self.rhs.dtype} vs {other.rhs.dtype}."
            )

    # -----------------------------------------------
    # --- linear equations methods ---
    # -----------------------------------------------

    def scalarize(self) -> list["LinearEqs"]:
        """Scalarize the vector equations."""
        mat_lst = self._mat.scalarize()
        rhs_lst = self._rhs.scalarize()

        eqs = []
        if len(mat_lst) == 1:
            for rhs in rhs_lst:
                eqs.append(LinearEqs(self.variable, self._mat, rhs))
        else:
            for mat, rhs in zip(mat_lst, rhs_lst):
                eqs.append(LinearEqs(self.variable, mat, rhs))
        return eqs

    def solve(self, method: str = "numpy") -> np.ndarray:
        """Solve the linear equations."""
        results = []
        for eq in self.scalarize():
            if method == "numpy":
                try:
                    # If the matrix is all zeros, the solution is the right-hand side.
                    if eq.matrix.nnz == 0:
                        results.append(eq.rhs.data)
                    else:
                        # Solve the linear equations using numpy.
                        results.append(
                            np.linalg.solve(eq.matrix.to_dense(), eq.rhs.data)
                        )
                except:
                    raise RuntimeError("Cannot solve linear equations.")
            elif method == "torch":
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                try:
                    # If the matrix is all zeros, the solution is the right-hand side.
                    if eq.matrix.nnz == 0:
                        results.append(eq.rhs.data)
                    else:
                        # Solve the linear equations using torch.
                        sparse_mat = eq.matrix.data.to_sparse_csr()
                        dense_vec = torch.from_numpy(eq.rhs.data).to(device)
                        results.append(
                            torch.sparse.spsolve(
                                sparse_mat,
                                dense_vec,
                            )
                            .cpu()
                            .numpy()
                        )
                except:
                    raise RuntimeError("Cannot solve linear equations.")
            else:
                raise ValueError(f"Unsupported algorithm {method}.")
        solution = np.array(results).T
        return solution
