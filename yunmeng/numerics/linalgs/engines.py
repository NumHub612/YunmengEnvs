# -*- encoding: utf-8 -*-
"""
Copyright (C) 2026, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Linear algebra solver engines.
"""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from typing import Literal

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu, spsolve

from yunmeng.interfaces.supports import IMatrix

try:
    import torch

    _HAS_TORCH = True
except ImportError:  # pragma: no cover
    torch = None
    _HAS_TORCH = False


# -----------------------------------------------
# region Engine base
# -----------------------------------------------


class LinearEngine(ABC):
    """Abstract base for linear system solvers: A @ x = b.

    All engines accept a single RHS (shape (N,)) or multi-RHS
    (shape (N, K)), and return the solution in the same layout.
    """

    #: Whether solve() preserves the autograd graph of the rhs
    #: (TRAIN contract). Engines using scipy / host round-trips are
    #: EVAL-only and set this to False.
    differentiable: bool = False

    @property
    @abstractmethod
    def name(self) -> str:
        """Canonical engine name."""
        raise NotImplementedError

    @abstractmethod
    def solve(self, mat_data, rhs):
        """Solve A @ x = b with backend-native data."""
        raise NotImplementedError


# -----------------------------------------------
# region Np/Sp engines (EVAL-only)
# -----------------------------------------------


class NumpyDirectSolver(LinearEngine):
    """Sparse direct solve via scipy.sparse.linalg.spsolve."""

    name = "numpy_direct"
    differentiable = False

    def solve(self, mat_sparse: sp.spmatrix, rhs_array: np.ndarray) -> np.ndarray:
        if rhs_array.ndim == 1 or (rhs_array.ndim == 2 and rhs_array.shape[1] == 1):
            return spsolve(mat_sparse, rhs_array.ravel())
        return NumpySparseLUSolver().solve(mat_sparse, rhs_array)


class NumpySparseLUSolver(LinearEngine):
    """Sparse LU factorization (splu): factor once, solve many RHS."""

    name = "numpy_lu"
    differentiable = False

    def solve(self, mat_sparse: sp.spmatrix, rhs_array: np.ndarray) -> np.ndarray:
        mat_csc = mat_sparse.tocsc() if mat_sparse.format != "csc" else mat_sparse
        return splu(mat_csc).solve(rhs_array)


# -----------------------------------------------
# region Torch engines
# -----------------------------------------------


class TorchDirectSolver(LinearEngine):
    """Dense direct solve via torch.linalg.solve (differentiable)."""

    name = "torch_direct"
    differentiable = True

    def solve(self, mat_tensor, rhs_tensor):
        was_1d = rhs_tensor.dim() == 1
        if was_1d:
            rhs_tensor = rhs_tensor.unsqueeze(-1)
        solution = torch.linalg.solve(mat_tensor, rhs_tensor)
        return solution.squeeze(-1) if was_1d else solution


class TorchSparseCGSolver(LinearEngine):
    """Sparse Conjugate Gradient for SPD systems.

    Single RHS solved directly; multi-RHS via loop. Falls back to
    dense direct solve below ``fallback_threshold`` or on CG
    non-convergence.
    """

    name = "torch_cg"
    differentiable = True

    def __init__(self, fallback_threshold: int = 500, maxiter: int = 1000):
        self._fallback_threshold = fallback_threshold
        self._maxiter = maxiter

    def solve(self, mat_tensor, rhs_tensor):
        was_1d = rhs_tensor.dim() == 1
        if was_1d:
            rhs_tensor = rhs_tensor.unsqueeze(-1)

        dense_solution = lambda b: torch.linalg.solve(  # noqa: E731
            mat_tensor.to_dense() if mat_tensor.is_sparse else mat_tensor, b
        )

        if mat_tensor.shape[0] < self._fallback_threshold:
            sol = dense_solution(rhs_tensor)
            return sol.squeeze(-1) if was_1d else sol

        try:
            from torch.sparse import linalg as sparse_linalg
        except ImportError:  # pragma: no cover
            warnings.warn(
                "torch.sparse.linalg not available, using dense fallback",
                RuntimeWarning,
            )
            sol = dense_solution(rhs_tensor)
            return sol.squeeze(-1) if was_1d else sol

        solutions = []
        for k in range(rhs_tensor.shape[1]):
            b_k = rhs_tensor[:, k]
            x_k, info = sparse_linalg.cg(mat_tensor, b_k, maxiter=self._maxiter)
            if info > 0:
                warnings.warn(
                    f"CG (rhs {k}) did not converge in {info} iterations, "
                    "falling back to dense direct solve",
                    RuntimeWarning,
                )
                x_k = dense_solution(b_k.unsqueeze(-1)).squeeze(-1)
            solutions.append(x_k)
        sol = torch.stack(solutions, dim=-1)
        return sol.squeeze(-1) if was_1d else sol


# -----------------------------------------------
# region Registry
# -----------------------------------------------

_ALGORITHM_REGISTRY: dict[str, type[LinearEngine]] = {}


def register_engine(name: str, engine_cls: type[LinearEngine]) -> None:
    """Register an engine under a canonical name."""
    _ALGORITHM_REGISTRY[name] = engine_cls


def get_engine(name: str) -> LinearEngine:
    """Instantiate a registered engine by name."""
    if name not in _ALGORITHM_REGISTRY:
        raise ValueError(
            f"Unknown engine: {name!r}. Available: {sorted(_ALGORITHM_REGISTRY)}"
        )
    return _ALGORITHM_REGISTRY[name]()


def select_engine(
    matrix: IMatrix,
    algorithm: Literal["auto", "direct", "cg", "lu"] = "auto",
) -> LinearEngine:
    """Select an engine for a matrix.

    Args:
        matrix: coefficient matrix; dispatch key is ``matrix.backend``
            ("numpy" | "torch"), not the concrete class.
        algorithm: "auto" chooses by backend/sparsity/size; "direct",
            "cg" (torch only), "lu" (numpy only) force a family.
    """
    backend = matrix.backend
    is_sparse = _is_sparse(matrix)

    if algorithm == "auto":
        if backend == "torch":
            return get_engine("torch_cg" if is_sparse else "torch_direct")
        if is_sparse and matrix.shape[0] > 100:
            return get_engine("numpy_lu")
        return get_engine("numpy_direct")

    if algorithm == "direct":
        return get_engine(f"{backend}_direct")
    if algorithm == "cg":
        if backend != "torch":
            raise ValueError("CG engine requires a torch-backed matrix")
        return get_engine("torch_cg")
    if algorithm == "lu":
        if backend != "numpy":
            raise ValueError("LU engine requires a numpy-backed matrix")
        return get_engine("numpy_lu")
    raise ValueError(f"Unknown algorithm: {algorithm!r}")


def _is_sparse(matrix: IMatrix) -> bool:
    data = matrix.data
    if sp.issparse(data):
        return True
    return bool(getattr(data, "is_sparse", False))


register_engine("numpy_direct", NumpyDirectSolver)
register_engine("numpy_lu", NumpySparseLUSolver)
register_engine("torch_direct", TorchDirectSolver)
register_engine("torch_cg", TorchSparseCGSolver)
