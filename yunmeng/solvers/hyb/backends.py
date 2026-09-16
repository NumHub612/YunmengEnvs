# -*- encoding: utf-8 -*-
"""
Copyright (C) 2026, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Concrete array backends: NumpyBackend (inference) and TorchBackend
(differentiable). Both satisfy the IBackend protocol.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from yunmeng.interfaces.types import ArrayLike, DeviceType

try:
    import torch

    _HAS_TORCH = True
except ImportError:  # pragma: no cover
    torch = None
    _HAS_TORCH = False


class NumpyBackend:
    """Pure-inference backend over numpy."""

    @property
    def name(self) -> str:
        return "numpy"

    @property
    def xp(self) -> object:
        return np

    @property
    def differentiable(self) -> bool:
        return False

    def asarray(
        self, data: object, dtype: str | None = None, device: DeviceType | None = None
    ) -> ArrayLike:
        return np.asarray(data, dtype=dtype or "float64")

    def zeros(
        self,
        shape: Sequence[int],
        dtype: str | None = None,
        device: DeviceType | None = None,
    ) -> ArrayLike:
        return np.zeros(tuple(shape), dtype=dtype or "float64")

    def zeros_like(self, a: ArrayLike) -> ArrayLike:
        return np.zeros_like(a)

    def to_device(self, a: ArrayLike, device: DeviceType) -> ArrayLike:
        return a

    def to_host(self, a: ArrayLike) -> ArrayLike:
        return np.asarray(a, dtype="float64")

    def where(self, cond: ArrayLike, x: ArrayLike, y: ArrayLike) -> ArrayLike:
        return np.where(cond, x, y)

    def maximum(self, x: ArrayLike, y: ArrayLike) -> ArrayLike:
        return np.maximum(x, y)

    def minimum(self, x: ArrayLike, y: ArrayLike) -> ArrayLike:
        return np.minimum(x, y)

    def concatenate(self, arrays: Sequence[ArrayLike], axis: int = 0) -> ArrayLike:
        return np.concatenate(list(arrays), axis=axis)

    def matmul(self, a: ArrayLike, b: ArrayLike) -> ArrayLike:
        return a @ b

    def einsum(self, equation: str, *operands: ArrayLike) -> ArrayLike:
        return np.einsum(equation, *operands)

    def scatter_add(
        self, target: ArrayLike, indices: ArrayLike, values: ArrayLike
    ) -> ArrayLike:
        out = np.array(target, copy=True)
        np.add.at(out, np.asarray(indices), np.asarray(values))
        return out

    def solve(self, a: ArrayLike, b: ArrayLike) -> ArrayLike:
        return np.linalg.solve(np.asarray(a), np.asarray(b))


class TorchBackend:
    """Differentiable backend over torch (float64 by default)."""

    def __init__(self, device: DeviceType = "cpu"):
        if not _HAS_TORCH:
            raise RuntimeError("torch is not available in this environment")
        self._device = torch.device(device if device != "auto" else "cpu")

    @property
    def name(self) -> str:
        return "torch"

    @property
    def xp(self) -> object:
        return torch

    @property
    def differentiable(self) -> bool:
        return True

    @property
    def device(self) -> "torch.device":
        return self._device

    def _torch_dtype(self, dtype: str | None) -> "torch.dtype":
        mapping = {
            "float64": torch.float64,
            "float32": torch.float32,
            "int64": torch.int64,
            "int32": torch.int32,
        }
        return mapping.get(dtype or "float64", torch.float64)

    def asarray(
        self, data: object, dtype: str | None = None, device: DeviceType | None = None
    ) -> ArrayLike:
        if isinstance(data, torch.Tensor):
            return data.to(
                dtype=self._torch_dtype(dtype), device=device or self._device
            )
        return torch.as_tensor(
            data, dtype=self._torch_dtype(dtype), device=device or self._device
        )

    def zeros(
        self,
        shape: Sequence[int],
        dtype: str | None = None,
        device: DeviceType | None = None,
    ) -> ArrayLike:
        return torch.zeros(
            tuple(shape), dtype=self._torch_dtype(dtype), device=device or self._device
        )

    def zeros_like(self, a: ArrayLike) -> ArrayLike:
        return torch.zeros_like(a)

    def to_device(self, a: ArrayLike, device: DeviceType) -> ArrayLike:
        return a.to(device)

    def to_host(self, a: ArrayLike) -> ArrayLike:
        if isinstance(a, torch.Tensor):
            return a.detach().cpu().numpy().astype("float64")
        return np.asarray(a, dtype="float64")

    def where(self, cond: ArrayLike, x: ArrayLike, y: ArrayLike) -> ArrayLike:
        return torch.where(cond, x, y)

    def maximum(self, x: ArrayLike, y: ArrayLike) -> ArrayLike:
        return torch.maximum(x, y)

    def minimum(self, x: ArrayLike, y: ArrayLike) -> ArrayLike:
        return torch.minimum(x, y)

    def concatenate(self, arrays: Sequence[ArrayLike], axis: int = 0) -> ArrayLike:
        return torch.cat(list(arrays), dim=axis)

    def matmul(self, a: ArrayLike, b: ArrayLike) -> ArrayLike:
        return a @ b

    def einsum(self, equation: str, *operands: ArrayLike) -> ArrayLike:
        return torch.einsum(equation, *operands)

    def scatter_add(
        self, target: ArrayLike, indices: ArrayLike, values: ArrayLike
    ) -> ArrayLike:
        idx = torch.as_tensor(indices, dtype=torch.long, device=target.device)
        out = target.clone()
        out.index_add_(0, idx, values)
        return out

    def solve(self, a: ArrayLike, b: ArrayLike) -> ArrayLike:
        # Dense differentiable solve (LU); torch.linalg.solve keeps the graph.
        return torch.linalg.solve(a, b)


def get_backend(name: str, device: DeviceType = "cpu"):
    """Factory: "numpy" | "torch"."""
    if name == "numpy":
        return NumpyBackend()
    if name == "torch":
        return TorchBackend(device=device)
    raise ValueError(f"unknown backend: {name!r}")
