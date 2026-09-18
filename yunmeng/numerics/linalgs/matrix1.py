# -*- encoding: utf-8 -*-
"""
Copyright (C) 2026, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Numpy matrix.
"""

from __future__ import annotations
import numpy as np
from typing import Union
import scipy.sparse as sp

from yunmeng.interfaces.types import ArrayLike, DeviceType
from yunmeng.interfaces.supports import IMatrix


class NumpyMatrix:
    """Scipy-sparse backed matrix. EVAL-only (no autograd)."""

    backend = "numpy"

    def __init__(self, sparse_matrix: sp.spmatrix):
        if not sp.issparse(sparse_matrix):
            raise TypeError("Input must be a scipy sparse matrix")
        self._data = sparse_matrix
        self._needs_csr = sparse_matrix.format != "csr"

    def _ensure_csr(self):
        if self._needs_csr:
            self._data = self._data.tocsr()
            self._needs_csr = False

    # -- factories ----------------------------------

    @classmethod
    def from_data(
        cls,
        values: ArrayLike,
        indices: ArrayLike = None,
        shape: tuple[int, int] = None,
        device: DeviceType = None,
    ) -> "NumpyMatrix":
        if device not in (None, "cpu"):
            raise ValueError("NumpyMatrix only supports CPU.")
        if indices is None:
            mat = sp.csr_matrix(values)
        else:
            row_idx, col_idx = indices
            if shape is None:
                shape = (int(np.max(row_idx)) + 1, int(np.max(col_idx)) + 1)
            mat = sp.coo_matrix((values, (row_idx, col_idx)), shape=shape).tocsr()
        return cls(mat)

    @classmethod
    def from_coo(
        cls,
        shape: tuple[int, int],
        values: ArrayLike,
        rows: ArrayLike,
        cols: ArrayLike,
        device: DeviceType = None,
    ) -> "NumpyMatrix":
        return cls.from_data(values, indices=(rows, cols), shape=shape)

    @classmethod
    def zeros(cls, shape: tuple[int, int], device: DeviceType = None) -> "NumpyMatrix":
        if device not in (None, "cpu"):
            raise ValueError("NumpyMatrix only supports CPU.")
        return cls(sp.dok_matrix(shape, dtype=np.float64))

    @classmethod
    def identity(cls, size: int, device: DeviceType = None) -> "NumpyMatrix":
        if device not in (None, "cpu"):
            raise ValueError("NumpyMatrix only supports CPU.")
        return cls(sp.eye(size, dtype=np.float64, format="csr"))

    # -- properties --------------------------------

    @property
    def data(self) -> sp.spmatrix:
        self._ensure_csr()
        return self._data

    @property
    def shape(self) -> tuple[int, int]:
        return self._data.shape

    @property
    def nnz(self) -> int:
        return self._data.nnz

    @property
    def device(self) -> DeviceType:
        return "cpu"

    @property
    def T(self) -> "NumpyMatrix":
        return NumpyMatrix(self._data.transpose())

    # -- utils -------------------------------------

    def to(self, device: DeviceType) -> "NumpyMatrix":
        if device != "cpu":
            raise ValueError("NumpyMatrix cannot move off CPU.")
        return self

    def to_dense(self) -> np.ndarray:
        return self._data.toarray()

    def convert(self, format: str) -> "NumpyMatrix":
        return NumpyMatrix(self._data.asformat(format))

    def diagonal(self, offset: int = 0) -> np.ndarray:
        return self._data.diagonal(k=offset)

    # -- element access ----------------------------

    def __getitem__(self, index: tuple):
        return self._data[index]

    def __setitem__(self, index: tuple, value: float):
        self._data[index] = value
        self._needs_csr = True

    # -- operators ---------------------------------

    def __matmul__(
        self, other: Union["NumpyMatrix", np.ndarray]
    ) -> Union["NumpyMatrix", np.ndarray]:
        if isinstance(other, NumpyMatrix):
            self._ensure_csr()
            other._ensure_csr()
            return NumpyMatrix(self._data @ other._data)
        if isinstance(other, np.ndarray):
            self._ensure_csr()
            return self._data @ other
        raise TypeError(f"Unsupported type for matmul: {type(other)}")

    def __add__(self, other: "NumpyMatrix") -> "NumpyMatrix":
        if not isinstance(other, NumpyMatrix):
            raise TypeError("Can only add NumpyMatrix to NumpyMatrix")
        return NumpyMatrix(self._data + other._data)

    def __sub__(self, other: "NumpyMatrix") -> "NumpyMatrix":
        if not isinstance(other, NumpyMatrix):
            raise TypeError("Can only subtract NumpyMatrix from NumpyMatrix")
        return NumpyMatrix(self._data - other._data)

    def __mul__(self, scalar: float) -> "NumpyMatrix":
        return NumpyMatrix(self._data * scalar)
