# -*- encoding: utf-8 -*-
"""
Copyright (C) 2026, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Torch matrix.
"""

from __future__ import annotations
import numpy as np
from typing import Union
import scipy.sparse as sp

from yunmeng.interfaces.types import ArrayLike, DeviceType
from yunmeng.interfaces.supports import IMatrix

try:
    import torch

    _HAS_TORCH = True
except ImportError:  # pragma: no cover
    torch = None
    _HAS_TORCH = False


class TorchMatrix:
    """PyTorch backed matrix (dense or sparse COO/CSR).

    Dense tensors support autograd end-to-end; sparse tensors are
    used with the CG engine, whose differentiability depends on
    torch.sparse support in the installed torch version.
    """

    backend = "torch"

    def __init__(self, tensor: "torch.Tensor"):
        self._data = tensor
        self._nnz_cache = None

    # -- factories -----------------------------

    @classmethod
    def from_data(
        cls,
        values: ArrayLike,
        indices: ArrayLike = None,
        shape: tuple[int, int] = None,
        device: DeviceType = None,
    ) -> "TorchMatrix":
        values = torch.as_tensor(values, dtype=torch.float64)
        if indices is not None:
            row_idx = torch.as_tensor(indices[0], dtype=torch.long)
            col_idx = torch.as_tensor(indices[1], dtype=torch.long)
            idx = torch.stack([row_idx, col_idx], dim=0)
            device = device or "cpu"
            idx = idx.to(device)
            values = values.to(device)
            if shape is None:
                shape = (int(idx[0].max()) + 1, int(idx[1].max()) + 1)
            coo = torch.sparse_coo_tensor(idx, values, size=shape, device=device)
            coo = coo.coalesce()
            if device.startswith("cuda"):
                try:
                    return cls(coo.to_sparse_csr())
                except Exception:  # pragma: no cover
                    pass
            return cls(coo)
        if shape is not None:
            values = values.reshape(shape)
        if device is not None:
            values = values.to(device)
        return cls(values)

    @classmethod
    def from_coo(
        cls,
        shape: tuple[int, int],
        values: ArrayLike,
        rows: ArrayLike,
        cols: ArrayLike,
        device: DeviceType = None,
    ) -> "TorchMatrix":
        return cls.from_data(values, indices=(rows, cols), shape=shape, device=device)

    @classmethod
    def from_csr(
        cls,
        shape: tuple[int, int],
        values: ArrayLike,
        ptrs: ArrayLike,
        idxs: ArrayLike,
        device: DeviceType = None,
    ) -> "TorchMatrix":
        device = device or "cpu"
        csr = torch.sparse_csr_tensor(
            torch.as_tensor(ptrs, dtype=torch.long).to(device),
            torch.as_tensor(idxs, dtype=torch.long).to(device),
            torch.as_tensor(values, dtype=torch.float64).to(device),
            size=shape,
            device=device,
        )
        return cls(csr)

    @classmethod
    def zeros(cls, shape: tuple[int, int], device: DeviceType = None) -> "TorchMatrix":
        device = device or "cpu"
        idx = torch.empty((2, 0), dtype=torch.long, device=device)
        vals = torch.empty(0, dtype=torch.float64, device=device)
        return cls(torch.sparse_coo_tensor(idx, vals, size=shape, device=device))

    @classmethod
    def identity(cls, size: int, device: DeviceType = None) -> "TorchMatrix":
        device = device or "cpu"
        i = torch.arange(size, device=device)
        idx = torch.stack([i, i], dim=0)
        vals = torch.ones(size, dtype=torch.float64, device=device)
        return cls(torch.sparse_coo_tensor(idx, vals, size=(size, size)).coalesce())

    # -- properties ----------------------------

    @property
    def data(self) -> "torch.Tensor":
        return self._data

    @property
    def shape(self) -> tuple[int, int]:
        return tuple(self._data.shape)

    @property
    def nnz(self) -> int:
        if self._nnz_cache is None:
            if self._data.is_sparse:
                self._nnz_cache = self._data.coalesce().values().shape[0]
            else:
                self._nnz_cache = int(torch.count_nonzero(self._data).item())
        return self._nnz_cache

    @property
    def device(self) -> DeviceType:
        return str(self._data.device)

    @property
    def T(self) -> "TorchMatrix":
        t = self._data.transpose(0, 1)
        if t.is_sparse:
            t = t.coalesce()
        return TorchMatrix(t)

    # -- utils ---------------------------------

    def to(self, device: DeviceType) -> "TorchMatrix":
        return TorchMatrix(self._data.to(device))

    def to_dense(self) -> "torch.Tensor":
        return self._data.to_dense() if self._data.is_sparse else self._data

    def convert(self, format: str) -> "TorchMatrix":
        if format == "csr":
            return (
                self
                if self._data.layout == torch.sparse_csr
                else TorchMatrix(self._data.to_sparse_csr())
            )
        if format == "coo":
            return (
                self
                if self._data.layout == torch.sparse_coo
                else TorchMatrix(self._data.to_sparse_coo())
            )
        raise RuntimeError(f"Format {format!r} not supported.")

    def diagonal(self, offset: int = 0) -> "torch.Tensor":
        if self._data.is_sparse:
            coo = self._data.coalesce()
            rows, cols, vals = coo.indices()[0], coo.indices()[1], coo.values()
            mask = (cols - rows) == offset
            return vals[mask]
        return torch.diag(self._data, diagonal=offset)

    # -- operators -----------------------------

    def __getitem__(self, index: tuple):
        return self._data[index]

    def __matmul__(
        self, other: Union["TorchMatrix", "torch.Tensor"]
    ) -> Union["TorchMatrix", "torch.Tensor"]:
        if isinstance(other, TorchMatrix):
            return TorchMatrix(self._data @ other._data)
        if isinstance(other, torch.Tensor):
            if other.dim() in (1, 2):
                return self._data @ other
            raise ValueError("Invalid tensor dimension")
        raise TypeError(f"Unsupported type: {type(other)}")

    def __add__(self, other: "TorchMatrix") -> "TorchMatrix":
        if not isinstance(other, TorchMatrix):
            raise TypeError("Addition requires TorchMatrix")
        return TorchMatrix(self._data + other._data)

    def __sub__(self, other: "TorchMatrix") -> "TorchMatrix":
        if not isinstance(other, TorchMatrix):
            raise TypeError("Subtraction requires TorchMatrix")
        return TorchMatrix(self._data - other._data)

    def __mul__(self, scalar: float) -> "TorchMatrix":
        return TorchMatrix(self._data * scalar)
