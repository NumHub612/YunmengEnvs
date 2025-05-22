# -*- encoding: utf-8 -*-
"""
Copyright (C) 2025, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Linear algebra class.
"""
from core.numerics.matrix import Matrix, SparseMatrix
from core.numerics.fields import Field, VariableType
from configs.settings import settings
import numpy as np
import torch
import cupy as cp
import scipy.sparse as sp
from scipy.sparse.linalg import cg as scipy_cg
from scipy.sparse.linalg import spsolve as scipy_spsolve
from scipy.sparse import dok_matrix
from cupyx.scipy.sparse import coo_matrix
from cupyx.scipy.sparse.linalg import spsolve as cupy_spsolve
from cupyx.scipy.sparse.linalg import cg as cupy_cg


class LinearEqs:
    """
    Linear equations solver.
    """

    def __init__(
        self, variable: str, mat: Matrix, rhs: Field, device: torch.device = None
    ):
        """Linear equations solver.

        Args:
            variable: The target variable.
            mat: The coefficient matrix of the linear equations.
            rhs: The right-hand side of the linear equations.
            device: The device.
        """
        self._device = device or settings.DEVICE
        if isinstance(self._device, str):
            self._device = torch.device(self._device)

        self._var = variable
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
    # --- static methods ---
    # -----------------------------------------------

    @staticmethod
    def zeros(
        variable: str,
        size: int,
        data_type: VariableType = VariableType.SCALAR,
        rhs_type: VariableType = VariableType.SCALAR,
        device: torch.device = None,
    ) -> "LinearEqs":
        """Create a linear equations with all elements set to zero."""
        mat = SparseMatrix.zeros((size, size), data_type, device)
        rhs = Field(size, "none", rhs_type, device)
        return LinearEqs(variable, mat, rhs, device)

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

    def __iadd__(self, other: "LinearEqs"):
        self._check_compatible(other)
        self._mat += other.matrix
        self._rhs += other.rhs
        return self

    def __sub__(self, other: "LinearEqs"):
        self._check_compatible(other)
        return LinearEqs(
            self.variable,
            self._mat - other.matrix,
            self._rhs - other.rhs,
        )

    def __rsub__(self, other: "LinearEqs"):
        self._check_compatible(other)
        return LinearEqs(
            self.variable,
            other.matrix - self._mat,
            other.rhs - self._rhs,
        )

    def __isub__(self, other: "LinearEqs"):
        self._check_compatible(other)
        self._mat -= other.matrix
        self._rhs -= other.rhs
        return self

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
    # --- linear equations methods ---
    # -----------------------------------------------

    def scalarize(self) -> list["LinearEqs"]:
        """Scalarize the vector equations."""
        mat_lst = self._mat.scalarize()
        rhs_lst = self._rhs.scalarize()

        eqs = []
        if len(mat_lst) == 1:
            for i, rhs in enumerate(rhs_lst):
                var = f"{self.variable}_{i}"
                eqs.append(LinearEqs(var, mat_lst[0], rhs))
        else:
            for mat, rhs, i in zip(mat_lst, rhs_lst, range(len(mat_lst))):
                var = f"{self.variable}_{i}"
                eqs.append(LinearEqs(var, mat, rhs))
        return eqs

    def solve(self, method: str = None) -> Field:
        """Solve the linear equations."""
        if method is None:
            if isinstance(self._mat, SparseMatrix):
                method = self._mat.backend
            else:
                method = "scipy"

        if settings.DEVICE == "cpu":
            method = "scipy"

        if method == "torch" or method == "scipy":
            solutions = self._solve_by_scipy()
        elif method == "numpy":  # NOTE: require back to dense np matrix.
            solutions = self._solve_by_numpy()
        elif method == "cupy":  # NOTE: require gpu.
            solutions = self._solve_by_cupy()
        elif method == "torch":  # NOTE: depend on custom compiled torch.
            solutions = self._solve_by_scipy()
        else:
            raise ValueError(f"Unsupported algorithm {method}.")

        result = Field(
            self.size,
            self._rhs.etype,
            self._rhs.dtype,
            solutions,
            self._var,
            self._device,
        )
        return result

    def _solve_by_numpy(self) -> np.ndarray:
        """Solve the linear equations using numpy."""
        solutions = []
        for eqs in self.scalarize():
            try:
                # If the matrix is all zeros, return the right-hand side.
                if eqs.matrix.nnz[0] == 0:
                    solutions.append(eqs.rhs.to_np().flatten())
                else:
                    res = np.linalg.solve(eqs.matrix.to_dense(), eqs.rhs.to_np())
                    solutions.append(res.flatten())
            except:
                raise RuntimeError("Can not solve linear equations.")
        return np.array(solutions).T

    def _solve_by_scipy(self) -> np.ndarray:
        """Solve the linear equations using scipy."""
        fptype = np.float64 if settings.FPTYPE == "fp64" else np.float32
        if settings.FPTYPE == "fp16":
            fptype = np.float16

        solutions = []
        for eqs in self.scalarize():
            try:
                # If the matrix is all zeros, return the right-hand side.
                if eqs.matrix.nnz[0] == 0:
                    solutions.append(eqs.rhs.to_np().flatten())
                else:
                    shape = eqs.matrix.shape
                    mat = eqs.matrix.data
                    if isinstance(mat, torch.Tensor):
                        indices = mat.indices().cpu().numpy()
                        data = mat.values().cpu().numpy()
                        mat = sp.coo_matrix(
                            (data, (indices[0], indices[1])),
                            shape=shape,
                            dtype=fptype,
                        )
                        mat = mat.tocsr()
                    elif isinstance(mat, coo_matrix):
                        rows = cp.asnumpy(mat.row)
                        cols = cp.asnumpy(mat.col)
                        data = cp.asnumpy(mat.data)
                        mat = sp.coo_matrix(
                            (data, (rows, cols)), shape=shape, dtype=fptype
                        )
                        mat = mat.tocsr()
                    elif isinstance(mat, dok_matrix):
                        mat = mat.tocsr()

                    b = eqs.rhs.to_np().flatten()
                    if shape[0] < 10_000:
                        res = scipy_spsolve(mat, b).flatten()
                    else:
                        tol, maxiter = settings.ITERATION
                        res = scipy_cg(
                            mat,
                            b,
                            tol=tol,
                            maxiter=maxiter,
                            atol=settings.TOLERANCE,
                        )[0].flatten()
                    solutions.append(res)
            except:
                raise RuntimeError("Cannot solve linear equations.")
        return np.array(solutions).T

    def _solve_by_torch(self) -> torch.Tensor:
        """Solve the linear equations using torch."""
        fptype = torch.float64 if settings.FPTYPE == "fp64" else torch.float32
        if settings.FPTYPE == "fp16":
            fptype = torch.float16

        solutions = []
        for eqs in self.scalarize():
            try:
                # If the matrix is all zeros, return the right-hand side.
                if eqs.matrix.nnz[0] == 0:
                    solutions.append(eqs.rhs.to_tensor(self._device).flatten())
                else:
                    shape = eqs.matrix.shape
                    mat = eqs.matrix.data
                    if isinstance(mat, coo_matrix):
                        data = torch.as_tensor(
                            mat.data, device=settings.DEVICE, dtype=fptype
                        )
                        indices = torch.as_tensor(
                            cp.vstack((mat.row, mat.col)), device=settings.DEVICE
                        )
                        coo = torch.sparse_coo_tensor(indices, data, shape)
                        mat = coo.to_sparse_csr()
                    elif isinstance(mat, dok_matrix):
                        indices = torch.as_tensor(
                            mat.nonzero(), device=settings.DEVICE, dtype=fptype
                        )
                        data = torch.tensor(
                            np.array(list(mat.values())),
                            dtype=fptype,
                            device=settings.DEVICE,
                        )
                        coo = torch.sparse_coo_tensor(indices, data, shape)
                        mat = coo.to_sparse_csr()

                    b = eqs.rhs.to_tensor(self._device)
                    res = torch.sparse.spsolve(mat, b)
                    solutions.append(res.flatten())
            except:
                raise RuntimeError("Cannot solve linear equations.")
        return torch.stack(solutions).T

    def _solve_by_cupy(self) -> cp.ndarray:
        """Solve the linear equations using cupy."""
        fptype = cp.float64 if settings.FPTYPE == "fp64" else cp.float32
        if settings.FPTYPE == "fp16":
            fptype = cp.float16

        solutions = []
        for eqs in self.scalarize():
            try:
                # If the matrix is all zeros, return the right-hand side.
                if eqs.matrix.nnz[0] == 0:
                    res = eqs.rhs.to_np().flatten()
                    arr = cp.asarray(res)
                    solutions.append(arr)
                else:
                    shape = eqs.matrix.shape
                    mat = eqs.matrix.data
                    if isinstance(mat, torch.Tensor):
                        indices = mat.indices().cpu().numpy()
                        rows = cp.asarray(indices[0])
                        cols = cp.asarray(indices[1])
                        data = cp.asarray(mat.values().cpu().numpy())
                        mat = coo_matrix(
                            (data, (rows, cols)),
                            shape=shape,
                            dtype=fptype,
                        )
                    elif isinstance(mat, dok_matrix):
                        indices = mat.nonzero()
                        data = np.array(list(mat.values()))
                        rows = cp.array(indices[0])
                        cols = cp.array(indices[1])
                        values = cp.array(data)
                        mat = coo_matrix(
                            (values, (rows, cols)), shape=shape, dtype=fptype
                        )

                    b = eqs.rhs.to_np().flatten()
                    b = cp.asarray(b)
                    if shape[0] < 10_000:
                        res = cupy_spsolve(mat, b).flatten()
                    else:
                        tol, maxiter = settings.ITERATION
                        res = cupy_cg(
                            mat,
                            b,
                            tol=tol,
                            maxiter=maxiter,
                            atol=settings.TOLERANCE,
                        )[0].flatten()
                    solutions.append(res)
            except:
                raise RuntimeError("Cannot solve linear equations.")
        result = cp.stack(solutions).T
        result = torch.as_tensor(result, device=self._device)  # convert to torch
        return result
