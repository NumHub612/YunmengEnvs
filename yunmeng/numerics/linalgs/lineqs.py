# -*- encoding: utf-8 -*-
"""
Copyright (C) 2025, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Linear equations container and interface.

This module defines LinearEqs — the high-level interface for linear systems.
Actual solving algorithms live in linalg_solvers.py and are selected
automatically based on matrix properties, or manually via algorithm= keyword.

Usage:
    eqs = LinearEqs(matrix, rhs_field)
    solution = eqs.solve()                    # auto-select solver
    solution = eqs.solve(algorithm="direct")  # force direct solve
    solution = eqs.solve(algorithm="cg")      # force CG
"""

from yunmeng.numerics.linalgs.matrixes import Matrix, TorchMatrix, NumpyMatrix
from yunmeng.numerics.fields import Field, FieldMeta, VariableType
from yunmeng.numerics.linalgs.engines import (
    select_engine,
    LinearEngine,
)

import numpy as np
import torch


class LinearEqs:
    """
    Linear equations container: A @ x = b.

    Responsibilities:
    - Hold matrix (A) and RHS field (b)
    - Provide algebraic operations (+, -, iadd, isub)
    - Provide scalarize() for component-wise decomposition
    - Delegate actual solving to LinearSolver implementations
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
        """The coefficient matrix."""
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
    # region scalarize
    # -----------------------------------------------

    def scalarize(self) -> list["LinearEqs"]:
        """Scalarize the vector equations into per-component systems."""
        rhs_lst = self._rhs.scalarize()
        eqs = [LinearEqs(self.matrix, rhs) for rhs in rhs_lst]
        return eqs

    # -----------------------------------------------
    # region solve
    # -----------------------------------------------

    def solve(self, algorithm: str = "auto") -> Field:
        """Solve the linear equations.

        Args:
            algorithm: Solver selection:
                "auto" — Choose based on matrix properties
                "direct" — Dense direct solve
                "cg" — Conjugate Gradient (torch only)
                "lu" — Sparse LU factorization (numpy only)

        Returns:
            Solution as a Field matching the RHS vtype.
        """
        field_meta = self._rhs.meta
        n_comp = field_meta.vtype.ncom

        if n_comp == 1:
            # Scalar field: direct single solve
            result = self._solve_single(self._rhs, algorithm)
            return self._build_solution_field(result, field_meta, self.rhs.mesh_shards)
        else:
            # Vector/Tensor field: batch solve all components
            results = self._solve_batched(n_comp, algorithm)
            return self._build_solution_field(results, field_meta, self.rhs.mesh_shards)

    # -----------------------------------------------
    # region Conversions
    # -----------------------------------------------

    def _rhs_to_array(self, rhs_field: Field) -> np.ndarray:
        """Extract RHS data as flat 1D array, with shape normalization."""
        arr = rhs_field.gather_to_host()
        if arr.ndim == 2 and arr.shape[1] == 1:
            arr = arr.reshape(-1)
        return arr

    def _solve_single(self, rhs_field: Field, algorithm: str):
        """Solve a single RHS (scalar field) via selected solver."""
        rhs_val = self._rhs_to_array(rhs_field)
        mat = self._mat

        # Select and invoke solver
        solver = select_engine(mat, algorithm=algorithm)

        if isinstance(mat, TorchMatrix):
            rhs_tensor = torch.as_tensor(rhs_val, device=mat.data.device)
            return solver.solve(mat.data, rhs_tensor)
        elif isinstance(mat, NumpyMatrix):
            return solver.solve(mat.data, rhs_val)
        else:
            raise TypeError(f"Unsupported matrix type: {type(mat)}")

    def _solve_batched(self, n_comp: int, algorithm: str):
        """Batched solve for vector field (multiple RHS).

        Collects all component RHS into a matrix B (shape N x n_comp),
        then solves A @ X = B in one call.
        """
        # Extract all component RHS arrays
        rhs_arrays = []
        if self._rhs.vtype.is_vector:
            full_arr = self._rhs.gather_to_host()  # (N, n_comp)
            for c in range(n_comp):
                rhs_arrays.append(full_arr[:, c])
        else:
            # Fallback: scalarize
            scalar_fields = self._rhs.scalarize()
            for f in scalar_fields:
                rhs_arrays.append(self._rhs_to_array(f))

        mat = self._mat
        solver = select_engine(mat, algorithm=algorithm)

        if isinstance(mat, TorchMatrix):
            rhs_tensor = torch.stack(
                [torch.as_tensor(a, device=mat.data.device) for a in rhs_arrays],
                dim=-1,
            )
            return solver.solve(mat.data, rhs_tensor)
        elif isinstance(mat, NumpyMatrix):
            rhs_matrix = np.column_stack(rhs_arrays)  # (N, n_comp)
            return solver.solve(mat.data, rhs_matrix)
        else:
            raise TypeError(f"Unsupported matrix type: {type(mat)}")

    def _build_solution_field(
        self, result, field_meta: "FieldMeta", mesh_shards
    ) -> Field:
        """Build solution Field from solver result."""
        # Convert torch tensor to numpy for from_array
        if isinstance(result, torch.Tensor):
            result = result.detach().cpu().numpy()

        n_comp = field_meta.vtype.ncom

        if n_comp == 1:
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
