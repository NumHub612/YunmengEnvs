# -*- encoding: utf-8 -*-
"""
Copyright (C) 2025, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Linear algebra solver engines.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Literal
import warnings

import numpy as np
import torch
import scipy.sparse as sp
from scipy.sparse.linalg import splu, spsolve

from yunmeng.numerics.linalgs.matrixes import Matrix, TorchMatrix, NumpyMatrix

# -----------------------------------------------
# region Registry & Selection
# -----------------------------------------------

_ALGORITHM_REGISTRY: dict[str, type[LinearEngine]] = {}


def register_solver(name: str, engine_cls: type[LinearEngine]):
    """Register a solver under a canonical name."""
    _ALGORITHM_REGISTRY[name] = engine_cls


def select_engine(
    matrix: Matrix,
    algorithm: Literal["auto", "direct", "cg", "lu"] = "auto",
) -> LinearEngine:
    """Select the best solver for a given matrix.

    Args:
        matrix: The coefficient matrix (determines backend)
        algorithm: Solver algorithm preference:
            - "auto": Choose based on matrix properties (sparse/dense, size)
            - "direct": Dense direct solve (LU/Cholesky)
            - "cg": Conjugate Gradient (sparse SPD)
            - "lu": Sparse LU factorization (scipy splu)

    Returns:
        A LinearSolver instance ready to use.
    """
    is_torch = isinstance(matrix, TorchMatrix)
    is_sparse = getattr(matrix.data, "is_sparse", False)

    if algorithm == "auto":
        if is_torch:
            if is_sparse:
                return TorchSparseCGSolver()
            return TorchDirectSolver()
        else:
            if is_sparse and matrix.shape[0] > 100:
                return NumpySparseLUSolver()
            return NumpyDirectSolver()

    if algorithm == "direct":
        return TorchDirectSolver() if is_torch else NumpyDirectSolver()
    elif algorithm == "cg":
        if not is_torch:
            raise ValueError("CG solver requires TorchMatrix")
        return TorchSparseCGSolver()
    elif algorithm == "lu":
        if is_torch:
            raise ValueError("LU solver requires NumpyMatrix")
        return NumpySparseLUSolver()
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")


# -----------------------------------------------
# region Linear Engine
# -----------------------------------------------


class LinearEngine(ABC):
    """Abstract base for linear system solvers: A @ x = b.

    All solvers support both single RHS (1D array) and multi-RHS (2D array).
    """

    @abstractmethod
    def solve(self, mat_data, rhs: np.ndarray | torch.Tensor):
        """Solve A @ x = b.

        Args:
            mat_data: Matrix representation (backend-specific)
            rhs: Right-hand side. Single: shape (N,) or (N, 1).
                 Multi: shape (N, K) with K > 1.

        Returns:
            Solution array, same backend/type as rhs.
            Single RHS → shape (N,)
            Multi RHS → shape (N, K)
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def name(self) -> str:
        """Canonical solver name."""
        raise NotImplementedError


# -----------------------------------------------
# region Torch Solvers
# -----------------------------------------------


class TorchDirectSolver(LinearEngine):
    """Dense direct solver via torch.linalg.solve.

    Supports multi-RHS natively (B matrix shape N x K).
    """

    name = "torch_direct"

    def solve(self, mat_tensor: torch.Tensor, rhs_tensor: torch.Tensor) -> torch.Tensor:
        was_1d = rhs_tensor.dim() == 1
        if was_1d:
            rhs_tensor = rhs_tensor.unsqueeze(-1)

        # mat_tensor is dense
        solution = torch.linalg.solve(mat_tensor, rhs_tensor)
        return solution.squeeze(-1) if was_1d else solution


class TorchSparseCGSolver(LinearEngine):
    """Sparse iterative solver: Conjugate Gradient via torch.sparse.linalg.cg.

    Best for large sparse SPD matrices. Single RHS only natively;
    multi-RHS handled via loop.

    Falls back to dense direct solve if CG fails or for small matrices.
    """

    name = "torch_cg"

    def __init__(self, fallback_threshold: int = 500):
        """Args:
        fallback_threshold: If matrix dim < threshold, use dense direct solve
            instead of CG (faster for small matrices).
        """
        self._fallback_threshold = fallback_threshold

    def solve(self, mat_tensor: torch.Tensor, rhs_tensor: torch.Tensor) -> torch.Tensor:
        was_1d = rhs_tensor.dim() == 1
        if was_1d:
            rhs_tensor = rhs_tensor.unsqueeze(-1)

        dim = mat_tensor.shape[0]

        # Fallback to dense direct for small matrices
        if dim < self._fallback_threshold:
            dense_mat = mat_tensor.to_dense()
            solution = torch.linalg.solve(dense_mat, rhs_tensor)
            return solution.squeeze(-1) if was_1d else solution

        # Sparse CG
        try:
            from torch.sparse import linalg as sparse_linalg

            if rhs_tensor.shape[1] == 1:
                # Single RHS: direct CG
                solution, info = sparse_linalg.cg(mat_tensor, rhs_tensor)
                if info > 0:
                    warnings.warn(
                        f"CG did not converge in {info} iterations, "
                        "falling back to dense direct solve",
                        RuntimeWarning,
                    )
                    dense_mat = mat_tensor.to_dense()
                    solution = torch.linalg.solve(dense_mat, rhs_tensor)
            else:
                # Multi-RHS: CG loop
                solutions = []
                for k in range(rhs_tensor.shape[1]):
                    sol_k, info = sparse_linalg.cg(
                        mat_tensor, rhs_tensor[:, k].unsqueeze(-1)
                    )
                    if info > 0:
                        warnings.warn(
                            f"CG component {k} did not converge, "
                            "falling back to dense solve for this component",
                            RuntimeWarning,
                        )
                        dense_mat = mat_tensor.to_dense()
                        sol_k = torch.linalg.solve(
                            dense_mat, rhs_tensor[:, k].unsqueeze(-1)
                        )
                    solutions.append(sol_k.squeeze(-1))
                solution = torch.stack(solutions, dim=-1)

            return solution.squeeze(-1) if was_1d else solution

        except ImportError:
            warnings.warn(
                "torch.sparse.linalg.cg not available, using dense fallback",
                RuntimeWarning,
            )
            dense_mat = mat_tensor.to_dense()
            solution = torch.linalg.solve(dense_mat, rhs_tensor)
            return solution.squeeze(-1) if was_1d else solution


# -----------------------------------------------
# region Numpy/SciPy Solvers
# -----------------------------------------------


class NumpyDirectSolver(LinearEngine):
    """Sparse direct solver via scipy.sparse.linalg.spsolve.

    For single RHS. Multi-RHS falls back to splu.
    """

    name = "numpy_direct"

    def solve(self, mat_sparse: sp.spmatrix, rhs_array: np.ndarray) -> np.ndarray:
        if rhs_array.ndim == 1:
            return spsolve(mat_sparse, rhs_array)
        elif rhs_array.ndim == 2 and rhs_array.shape[1] == 1:
            return spsolve(mat_sparse, rhs_array.ravel())
        else:
            # Multi-RHS: delegate to LU solver
            return NumpySparseLUSolver().solve(mat_sparse, rhs_array)


class NumpySparseLUSolver(LinearEngine):
    """Sparse LU factorization via scipy.sparse.linalg.splu.

    Best for multi-RHS problems: factor once, solve times.
    """

    name = "numpy_lu"

    def solve(self, mat_sparse: sp.spmatrix, rhs_array: np.ndarray) -> np.ndarray:
        # Convert to CSC for efficient LU factorization
        mat_csc = mat_sparse.tocsc() if mat_sparse.format != "csc" else mat_sparse
        lu = splu(mat_csc)
        return lu.solve(rhs_array)


# -----------------------------------------------
# region Future solver stubs (extensibility)
# -----------------------------------------------


class TorchGMRESSolver(LinearEngine):
    """GMRES solver for non-symmetric sparse systems (Torch backend).

    TODO: Implement using iterative wrapper or external library.
    """

    name = "torch_gmres"

    def solve(self, mat_data, rhs):
        raise NotImplementedError("GMRES solver not yet implemented for torch backend")


class JacobiSolver(LinearEngine):
    """Jacobi iterative solver.

    TODO: Implement as a simple baseline iterative solver.
    Useful for preconditioning and GPU-friendly implementations.
    """

    name = "jacobi"

    def solve(self, mat_data, rhs):
        raise NotImplementedError("Jacobi solver not yet implemented")


# ---------------------------------------------------------------------------
# region Register all solvers
# ---------------------------------------------------------------------------

register_solver("torch_direct", TorchDirectSolver)
register_solver("torch_cg", TorchSparseCGSolver)
register_solver("numpy_direct", NumpyDirectSolver)
register_solver("numpy_lu", NumpySparseLUSolver)
