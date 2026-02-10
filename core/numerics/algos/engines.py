# -*- encoding: utf-8 -*-
"""
Copyright (C) 2026, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Linear algebra class.
"""
from core.numerics.enums import EngineMethod
from core.numerics.mats import Matrix
from core.numerics.fields import Field
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
from typing import Callable


def get_engine_method(method: str) -> Callable:
    """Get the engine method."""
    try:
        engine_method = EngineMethod.from_str(method)
        if engine_method == EngineMethod.NUMPY:
            return solve_by_numpy
        elif engine_method == EngineMethod.SCIPY:
            return solve_by_scipy
        elif engine_method == EngineMethod.CUPY:
            return solve_by_cupy
        elif engine_method == EngineMethod.TORCH:
            return solve_by_torch
    except ValueError:
        raise ValueError(f"Invalid engine method: {method}")


def solve_by_numpy(matrix: Matrix, rhs: Field) -> np.ndarray:
    """Solve the linear equations using numpy."""
    try:
        # If the matrix is all zeros, return the right-hand side.
        if matrix.nnz[0] == 0:
            return rhs.data.as_numpy().flatten()
        else:
            result = np.linalg.solve(matrix.to_dense(), rhs.data.as_numpy())
            return result.flatten()
    except:
        raise RuntimeError("Can not solve linear equations.")


def solve_by_scipy(matrix: Matrix, rhs: Field) -> np.ndarray:
    """Solve the linear equations using scipy."""
    try:
        # If the matrix is all zeros, return the right-hand side.
        if matrix.nnz[0] == 0:
            return rhs.to_np().flatten()
        else:
            b = rhs.to_np().flatten()
            shape = matrix.shape
            mat = matrix.data
            if isinstance(mat, torch.Tensor):
                indices = mat.indices().cpu().numpy()
                data = mat.values().cpu().numpy()
                mat = sp.coo_matrix(
                    (data, (indices[0], indices[1])),
                    shape=shape,
                    dtype=np.float64,
                )
                mat = mat.tocsr()
            elif isinstance(mat, coo_matrix):
                rows = cp.asnumpy(mat.row)
                cols = cp.asnumpy(mat.col)
                data = cp.asnumpy(mat.data)
                mat = sp.coo_matrix((data, (rows, cols)), shape=shape, dtype=np.float64)
                mat = mat.tocsr()
            elif isinstance(mat, dok_matrix):
                mat = mat.tocsr()

            if shape[0] < 10_000:
                return scipy_spsolve(mat, b).flatten()
            else:
                tol, maxiter = 1.0e-6, 1000
                return scipy_cg(
                    mat,
                    b,
                    tol=tol,
                    maxiter=maxiter,
                    atol=1.0e-6,
                )[0].flatten()
    except:
        raise RuntimeError("Cannot solve linear equations.")


def solve_by_torch(matrix: Matrix, rhs: Field) -> torch.Tensor:
    """Solve the linear equations using torch."""
    try:
        # If the matrix is all zeros, return the right-hand side.
        if matrix.nnz[0] == 0:
            return rhs.to_tensor().flatten()
        else:
            b = rhs.to_tensor()
            shape = matrix.shape
            mat = matrix.data
            if isinstance(mat, coo_matrix):
                data = torch.as_tensor(mat.data, dtype=torch.float64)
                indices = torch.as_tensor(cp.vstack((mat.row, mat.col)))
                coo = torch.sparse_coo_tensor(indices, data, shape)
                mat = coo.to_sparse_csr()
            elif isinstance(mat, dok_matrix):
                indices = torch.as_tensor(mat.nonzero(), dtype=torch.float64)
                data = torch.tensor(
                    np.array(list(mat.values())),
                    dtype=torch.float64,
                )
                coo = torch.sparse_coo_tensor(indices, data, shape)
                mat = coo.to_sparse_csr()

            result = torch.sparse.spsolve(mat, b)
            return result.flatten()
    except:
        raise RuntimeError("Cannot solve linear equations.")


def solve_by_cupy(matrix: Matrix, rhs: Field) -> cp.ndarray:
    """Solve the linear equations using cupy."""
    try:
        # If the matrix is all zeros, return the right-hand side.
        if matrix.nnz[0] == 0:
            res = rhs.to_np().flatten()
            return cp.asarray(res)
        else:
            b = cp.asarray(rhs.to_np().flatten())
            shape = matrix.shape
            mat = matrix.data
            if isinstance(mat, torch.Tensor):
                indices = mat.indices().cpu().numpy()
                rows = cp.asarray(indices[0])
                cols = cp.asarray(indices[1])
                data = cp.asarray(mat.values().cpu().numpy())
                mat = coo_matrix(
                    (data, (rows, cols)),
                    shape=shape,
                    dtype=cp.float64,
                )
            elif isinstance(mat, dok_matrix):
                indices = mat.nonzero()
                data = np.array(list(mat.values()))
                rows = cp.array(indices[0])
                cols = cp.array(indices[1])
                values = cp.array(data)
                mat = coo_matrix((values, (rows, cols)), shape=shape, dtype=cp.float64)

            if shape[0] < 10_000:
                return cupy_spsolve(mat, b).flatten()
            else:
                tol, maxiter = 1.0e-6, 1000
                return cupy_cg(
                    mat,
                    b,
                    tol=tol,
                    maxiter=maxiter,
                    atol=1.0e-6,
                )[0].flatten()
    except:
        raise RuntimeError("Cannot solve linear equations.")
