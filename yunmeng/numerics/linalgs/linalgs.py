# -*- encoding: utf-8 -*-
"""
Copyright (C) 2025, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Linear equations solver.
"""

from yunmeng.numerics.linalgs.matrixes import Matrix, TorchMatrix, NumpyMatrix
from yunmeng.numerics.fields import Field, FieldMeta, VariableType
import torch
import numpy as np


class LinearEqs:
    """
    Linear equations solver with batched multi-RHS support.
    """

    def __init__(self, mat: Matrix, rhs: Field):
        self._mat = mat
        self._rhs = rhs

        if mat.shape[0] != mat.shape[1] or mat.shape[0] != rhs.size:
            raise ValueError(
                f"The matrix is not square: {mat.shape}, or not compatible "
                f"with the rhs: {rhs.size}."
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
                f"Invalid LinearEqs operation with different sizes: "
                f"{self.size} vs {other.size}."
            )
        if other.rhs.vtype != self.rhs.vtype:
            raise ValueError(
                f"Invalid LinearEqs operation with different types: "
                f"{self.rhs.vtype} vs {other.rhs.vtype}."
            )

    def reset_rhs(self, rhs: Field):
        """Reset the right-hand side vector."""
        if rhs.size != self.size:
            raise ValueError(f"Invalid rhs size: {rhs.size} vs {self.size}.")
        self._rhs = rhs

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
        field_meta = self._rhs.meta
        n_comp = field_meta.vtype.ncom

        if n_comp == 1:
            # --- Scalar field: direct solve, no scalarize needed ---
            result = self._solve_single(self._rhs)
            return self._build_solution_field(result, field_meta, self.rhs.mesh_shards)
        else:
            # --- Vector/Tensor field: batch solve all components at once ---
            results = self._solve_batched(n_comp)
            return self._build_solution_field(results, field_meta, self.rhs.mesh_shards)

    def _rhs_to_array(self, rhs_field: Field) -> np.ndarray:
        """Extract RHS data as flat 1D array, with shape normalization."""
        arr = rhs_field.gather_to_host()
        # Normalize (N, 1) scalar shape to (N,)
        if arr.ndim == 2 and arr.shape[1] == 1:
            arr = arr.reshape(-1)
        return arr

    def _solve_single(self, rhs_field: Field):
        """Solve a single RHS (scalar field).

        Returns: np.ndarray or torch.Tensor (1D, shape (N,))
        """
        rhs_val = self._rhs_to_array(rhs_field)
        mat = self._mat

        if isinstance(mat, TorchMatrix):
            rhs_tensor = torch.as_tensor(rhs_val, device=mat.data.device)
            return self._solve_torch(mat.data, rhs_tensor)
        elif isinstance(mat, NumpyMatrix):
            return self._solve_numpy(mat.data, rhs_val)
        else:
            raise TypeError(f"Unsupported matrix type: {type(mat)}")

    def _solve_batched(self, n_comp: int):
        """Batched solve for vector field (multiple RHS).

        Collects all component RHS into a matrix B (shape N x n_comp),
        then solves A @ X = B in one call.

        Returns: np.ndarray or torch.Tensor (shape (N, n_comp))
        """
        # Extract all component RHS arrays
        rhs_arrays = []
        if self._rhs.vtype.is_vector:
            # Vector field: gather and split components
            full_arr = self._rhs.gather_to_host()  # (N, n_comp)
            for c in range(n_comp):
                rhs_arrays.append(full_arr[:, c])
        else:
            # Fallback: scalarize
            scalar_fields = self._rhs.scalarize()
            for f in scalar_fields:
                rhs_arrays.append(self._rhs_to_array(f))

        mat = self._mat

        if isinstance(mat, TorchMatrix):
            # Stack into (N, n_comp) matrix
            rhs_tensor = torch.stack(
                [torch.as_tensor(a, device=mat.data.device) for a in rhs_arrays],
                dim=-1,  # (N, n_comp)
            )
            return self._solve_torch(mat.data, rhs_tensor)
        elif isinstance(mat, NumpyMatrix):
            rhs_matrix = np.column_stack(rhs_arrays)  # (N, n_comp)
            return self._solve_numpy(mat.data, rhs_matrix)
        else:
            raise TypeError(f"Unsupported matrix type: {type(mat)}")

    def _build_solution_field(
        self, result, field_meta: "FieldMeta", mesh_shards
    ) -> Field:
        """Build solution Field from solver result.

        Args:
            result: np.ndarray or torch.Tensor
                Scalar: shape (N,) or (N, 1)
                Vector: shape (N, n_comp)
            field_meta: metadata from original RHS field
            mesh_shards: mesh shards for new field
        """
        # Convert torch tensor to numpy for from_array
        if isinstance(result, torch.Tensor):
            result = result.detach().cpu().numpy()

        n_comp = field_meta.vtype.ncom

        if n_comp == 1:
            # Scalar: ensure shape (N,)
            if result.ndim > 1:
                result = result.squeeze(-1)
            return Field.from_array(
                result,
                mesh_shards,
                VariableType.scalar(),
                field_meta.etype,
                field_meta.requires_grad,
            )
        else:
            # Vector: result shape (N, n_comp)
            if result.ndim == 1:
                result = result.reshape(-1, 1)
            dim = result.shape[1] if result.ndim > 1 else 1
            return Field.from_array(
                result,
                mesh_shards,
                VariableType.vector(dim),
                field_meta.etype,
                field_meta.requires_grad,
            )

    def _solve_torch(
        self, mat_tensor: torch.Tensor, rhs_tensor: torch.Tensor
    ) -> torch.Tensor:
        """Internal solver for PyTorch matrices.

        Supports both single RHS (shape N or Nx1) and multi-RHS (shape NxK).
        """
        # Ensure RHS is 2D: (N, K) where K is number of RHS
        was_1d = rhs_tensor.dim() == 1
        if was_1d:
            rhs_tensor = rhs_tensor.unsqueeze(-1)  # (N, 1)

        if mat_tensor.is_sparse:
            try:
                # Attempt sparse CG (Conjugate Gradient)
                from torch.sparse import linalg as sparse_linalg

                # CG handles single RHS only; loop for multi-RHS
                if rhs_tensor.shape[1] == 1:
                    solution, _ = sparse_linalg.cg(mat_tensor, rhs_tensor)
                else:
                    solutions = []
                    for k in range(rhs_tensor.shape[1]):
                        sol_k, _ = sparse_linalg.cg(
                            mat_tensor, rhs_tensor[:, k].unsqueeze(-1)
                        )
                        solutions.append(sol_k.squeeze(-1))
                    solution = torch.stack(solutions, dim=-1)
                if was_1d:
                    solution = solution.squeeze(-1)
                return solution
            except Exception:
                # Fallback: convert to dense
                dense_mat = mat_tensor.to_dense()
                solution = torch.linalg.solve(dense_mat, rhs_tensor)
                if was_1d:
                    solution = solution.squeeze(-1)
                return solution.squeeze(-1) if was_1d else solution
        else:
            # Dense matrix: torch.linalg.solve supports multi-RHS natively
            solution = torch.linalg.solve(mat_tensor, rhs_tensor)
            return solution.squeeze(-1) if was_1d else solution

    def _solve_numpy(self, mat_sparse, rhs_array: np.ndarray) -> np.ndarray:
        """Internal solver for Scipy/Numpy matrices.

        Supports both single RHS (shape N or Nx1) and multi-RHS (shape NxK).
        """
        from scipy.sparse.linalg import splu, spsolve

        if rhs_array.ndim == 1:
            # Single RHS
            return spsolve(mat_sparse, rhs_array)
        elif rhs_array.ndim == 2 and rhs_array.shape[1] == 1:
            # Single RHS in column form
            return spsolve(mat_sparse, rhs_array.ravel())
        else:
            # Multi-RHS: use splu for efficiency
            # Convert to CSC for splu (more efficient factorization)
            mat_csc = mat_sparse.tocsc() if mat_sparse.format != "csc" else mat_sparse
            lu = splu(mat_csc)
            return lu.solve(rhs_array)
