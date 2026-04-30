# -*- encoding: utf-8 -*-
"""
Copyright (C) 2025, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Linear algebra class.
"""
from yunmeng.numerics.mats.matrix import Matrix
from yunmeng.numerics.mats.sparse import TorchMatrix, NumpyMatrix
from yunmeng.numerics.fields.fields import Field, BackendType, get_backend
import torch
import scipy as sp
import numpy as np


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
        self._size = rhs.size

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

    @property
    def rhs(self) -> Field:
        """The right-hand side vector."""
        return self._rhs

    # -----------------------------------------------
    # region operations
    # -----------------------------------------------

    def __add__(self, other: "LinearEqs"):
        self._check_compatible(other)
        return LinearEqs(
            self._mat + other.matrix,
            self._rhs + other.rhs,
        )

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

    def __isub__(self, other: "LinearEqs"):
        self._check_compatible(other)
        self._mat -= other.matrix
        self._rhs -= other.rhs
        return self

    def _check_compatible(self, other: "LinearEqs"):
        if not isinstance(other, LinearEqs):
            raise ValueError(f"Require LinearEqs type, got {type(other)}.")
        if other.size != self.size:
            raise ValueError(
                f"Invalid LinearEqs operation with different sizes: \
                    {self.size} vs {other.size}."
            )
        if other.rhs.vtype != self.rhs.vtype:
            raise ValueError(
                f"Invalid LinearEqs operation with different types: \
                    {self.rhs.vtype} vs {other.rhs.vtype}."
            )

    # -----------------------------------------------
    # region solve
    # -----------------------------------------------

    def scalarize(self) -> list["LinearEqs"]:
        """Scalarize the vector equations."""
        rhs_lst = self._rhs.scalarize()
        eqs = [LinearEqs(self.matrix, rhs) for rhs in rhs_lst]
        return eqs

    def solve(self) -> Field:
        """Solve the linear equations."""
        results = []
        for eq in self.scalarize():
            mat = eq.matrix
            rhs_val = eq.rhs.gather_to_host()

            if isinstance(mat, TorchMatrix):
                rhs_val = torch.as_tensor(rhs_val, device=mat.data.device)
                result = self._solve_torch(mat.data, rhs_val)
            elif isinstance(mat, NumpyMatrix):
                result = self._solve_numpy(mat.data, rhs_val)
            else:
                raise TypeError(f"Unsupported matrix type: {type(mat)}")

            if result.ndim > 1:
                result = result.squeeze(-1)
            results.append(result)

        field_meta = self._rhs.meta
        back = get_backend(field_meta.btype)
        values = back.stack(results, axis=0)
        if field_meta.btype == BackendType.TORCH:
            values = values.moveaxis(0, -1)
        else:  # numpy
            values = np.moveaxis(values, 0, -1)
        return Field.from_array(
            values,
            self.rhs.mesh_shards,
            field_meta.vtype,
            field_meta.etype,
            field_meta.requires_grad,
        )

    def _solve_torch(
        self, mat_tensor: torch.Tensor, rhs_tensor: torch.Tensor
    ) -> torch.Tensor:
        """Internal solver for PyTorch matrices."""
        # Ensure RHS is correct shape (N, 1) or (N,)
        if rhs_tensor.dim() == 1:
            rhs_tensor = rhs_tensor.unsqueeze(-1)

        if mat_tensor.is_sparse:
            try:
                # Attempt sparse CG (Conjugate Gradient)
                from torch.sparse import linalg as sparse_linalg

                solution, _ = sparse_linalg.cg(mat_tensor, rhs_tensor)
                return solution.squeeze(-1)
            except Exception:
                # Final fallback
                dense_mat = mat_tensor.to_dense()
                solution = torch.linalg.solve(dense_mat, rhs_tensor)
                return solution.squeeze(-1)
        else:  # Dense matrix
            return torch.linalg.solve(
                mat_tensor,
                rhs_tensor,
            ).squeeze(-1)

    def _solve_numpy(self, mat_sparse, rhs_array: np.ndarray) -> np.ndarray:
        """Internal solver for Scipy/Numpy matrices."""
        from scipy.sparse.linalg import splu, spsolve

        # Ensure RHS is correct shape
        if rhs_array.ndim == 1:
            # spsolve handles 1D rhs
            return spsolve(mat_sparse, rhs_array)
        else:
            # For multiple RHS, splu is more efficient
            lu = splu(mat_sparse)
            return lu.solve(rhs_array)
