# -*- encoding: utf-8 -*-
"""
Copyright (C) 2026, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Concrete array backends: NumpyBackend (inference) and TorchBackend
(differentiable).
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from yunmeng.interfaces.types import ArrayLike, DeviceType
from yunmeng.interfaces.supports import IBackend

try:
    import torch

    _HAS_TORCH = True
except ImportError:  # pragma: no cover
    torch = None
    _HAS_TORCH = False

# ---------------------------------------------------
# region NumpyBackend
# ---------------------------------------------------


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

    # -- construction / conversion ------------------

    def asarray(
        self, data: object, dtype: str = None, device: DeviceType = None
    ) -> ArrayLike:
        if isinstance(data, np.ndarray) and dtype is None:
            return data
        return np.asarray(data, dtype=dtype or "float64")

    def zeros(
        self, shape: Sequence[int], dtype: str = None, device: DeviceType = None
    ) -> ArrayLike:
        return np.zeros(tuple(shape), dtype=dtype or "float64")

    def zeros_like(self, a: ArrayLike) -> ArrayLike:
        return np.zeros_like(a)

    def full(
        self,
        shape: Sequence[int],
        fill_value: float,
        dtype: str = None,
        device: DeviceType = None,
    ) -> ArrayLike:
        return np.full(
            tuple(shape),
            fill_value,
            dtype=dtype or "float64",
        )

    def eye(self, n: int, dtype: str = None) -> ArrayLike:
        return np.eye(n, dtype=dtype or "float64")

    def arange(
        self,
        n: int,
        dtype: str = None,
        device: DeviceType = None,
    ) -> ArrayLike:
        return np.arange(n, dtype=dtype or "int64")

    def stack(
        self,
        arrays: Sequence[ArrayLike],
        axis: int = 0,
    ) -> ArrayLike:
        return np.stack(list(arrays), axis=axis)

    def to_device(self, a: ArrayLike, device: DeviceType) -> ArrayLike:
        return a

    def to_host(self, a: ArrayLike) -> ArrayLike:
        return np.asarray(a, dtype="float64")

    # -- elementwise / selection --------------------

    def where(
        self,
        cond: ArrayLike,
        x: ArrayLike,
        y: ArrayLike,
    ) -> ArrayLike:
        return np.where(cond, x, y)

    def maximum(self, x: ArrayLike, y: ArrayLike) -> ArrayLike:
        return np.maximum(x, y)

    def minimum(self, x: ArrayLike, y: ArrayLike) -> ArrayLike:
        return np.minimum(x, y)

    def concatenate(
        self,
        arrays: Sequence[ArrayLike],
        axis: int = 0,
    ) -> ArrayLike:
        return np.concatenate(list(arrays), axis=axis)

    def sum(self, a: ArrayLike, axis: int = None) -> ArrayLike:
        return np.sum(a, axis=axis)

    def norm(self, a: ArrayLike) -> ArrayLike:
        return float(np.linalg.norm(a))

    # -- algorithmic primitives ---------------------

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

    def gather(
        self,
        a: ArrayLike,
        indices: ArrayLike,
        axis: int = 0,
    ) -> ArrayLike:
        return np.take(a, np.asarray(indices), axis=axis)

    def roll(self, a: ArrayLike, shift: int, axis: int) -> ArrayLike:
        return np.roll(a, shift, axis=axis)

    def pad(
        self,
        a: ArrayLike,
        pad_width: Sequence[tuple[int, int]],
        mode: str = "constant",
        value: float = 0.0,
    ) -> ArrayLike:
        if mode == "constant":
            return np.pad(a, pad_width, mode=mode, constant_values=value)
        return np.pad(a, pad_width, mode=mode)

    def solve(self, a: ArrayLike, b: ArrayLike) -> ArrayLike:
        return np.linalg.solve(np.asarray(a), np.asarray(b))


# ---------------------------------------------------
# region TorchBackend
# ---------------------------------------------------


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

    def _torch_dtype(self, dtype: str) -> "torch.dtype":
        mapping = {
            "float64": torch.float64,
            "float32": torch.float32,
            "int64": torch.int64,
            "int32": torch.int32,
        }
        return mapping.get(dtype or "float64", torch.float64)

    # -- construction / conversion ------------------

    def asarray(
        self, data: object, dtype: str = None, device: DeviceType = None
    ) -> ArrayLike:
        if isinstance(data, torch.Tensor):
            return data.to(
                dtype=self._torch_dtype(dtype),
                device=device or self._device,
            )
        return torch.as_tensor(
            data,
            dtype=self._torch_dtype(dtype),
            device=device or self._device,
        )

    def zeros(
        self,
        shape: Sequence[int],
        dtype: str = None,
        device: DeviceType = None,
    ) -> ArrayLike:
        return torch.zeros(
            tuple(shape),
            dtype=self._torch_dtype(dtype),
            device=device or self._device,
        )

    def zeros_like(self, a: ArrayLike) -> ArrayLike:
        return torch.zeros_like(a)

    def full(
        self,
        shape: Sequence[int],
        fill_value: float,
        dtype: str = None,
        device: DeviceType = None,
    ) -> ArrayLike:
        return torch.full(
            tuple(shape),
            fill_value,
            dtype=self._torch_dtype(dtype),
            device=device or self._device,
        )

    def eye(self, n: int, dtype: str = None) -> ArrayLike:
        return torch.eye(
            n,
            dtype=self._torch_dtype(dtype),
            device=self._device,
        )

    def arange(
        self,
        n: int,
        dtype: str = None,
        device: DeviceType = None,
    ) -> ArrayLike:
        return torch.arange(
            n,
            dtype=self._torch_dtype(dtype or "int64"),
            device=device or self._device,
        )

    def stack(self, arrays: Sequence[ArrayLike], axis: int = 0) -> ArrayLike:
        return torch.stack(list(arrays), dim=axis)

    def to_device(self, a: ArrayLike, device: DeviceType) -> ArrayLike:
        return a.to(device)

    def to_host(self, a: ArrayLike) -> ArrayLike:
        if isinstance(a, torch.Tensor):
            return a.detach().cpu().numpy().astype("float64")
        return np.asarray(a, dtype="float64")

    # -- elementwise / selection --------------------

    def where(self, cond: ArrayLike, x: ArrayLike, y: ArrayLike) -> ArrayLike:
        return torch.where(cond, x, y)

    def maximum(self, x: ArrayLike, y: ArrayLike) -> ArrayLike:
        return torch.maximum(x, y)

    def minimum(self, x: ArrayLike, y: ArrayLike) -> ArrayLike:
        return torch.minimum(x, y)

    def concatenate(
        self,
        arrays: Sequence[ArrayLike],
        axis: int = 0,
    ) -> ArrayLike:
        return torch.cat(list(arrays), dim=axis)

    def sum(self, a: ArrayLike, axis: int = None) -> ArrayLike:
        return torch.sum(a, dim=axis)

    def norm(self, a: ArrayLike) -> ArrayLike:
        return torch.linalg.norm(a)

    # -- algorithmic primitives ---------------------

    def matmul(self, a: ArrayLike, b: ArrayLike) -> ArrayLike:
        return a @ b

    def einsum(self, equation: str, *operands: ArrayLike) -> ArrayLike:
        return torch.einsum(equation, *operands)

    def scatter_add(
        self, target: ArrayLike, indices: ArrayLike, values: ArrayLike
    ) -> ArrayLike:
        idx = torch.as_tensor(
            indices,
            dtype=torch.long,
            device=target.device,
        )
        out = target.clone()
        out.index_add_(0, idx, values)
        return out

    def gather(
        self,
        a: ArrayLike,
        indices: ArrayLike,
        axis: int = 0,
    ) -> ArrayLike:
        idx = torch.as_tensor(indices, dtype=torch.long, device=a.device)
        return torch.index_select(a, axis, idx)

    def roll(self, a: ArrayLike, shift: int, axis: int) -> ArrayLike:
        return torch.roll(a, shift, dims=axis)

    def pad(
        self,
        a: ArrayLike,
        pad_width: Sequence[tuple[int, int]],
        mode: str = "constant",
        value: float = 0.0,
    ) -> ArrayLike:
        # torch.nn.functional.pad wants (last_dim_before, last_dim_after, ...)
        import torch.nn.functional as F

        flat: list[int] = []
        for before, after in reversed(list(pad_width)):
            flat.extend([before, after])
        if mode == "constant":
            return F.pad(a, flat, mode=mode, value=value)
        return F.pad(a, flat, mode=mode)

    def solve(self, a: ArrayLike, b: ArrayLike) -> ArrayLike:
        # Dense differentiable solve (LU); torch.linalg.solve keeps the graph.
        return torch.linalg.solve(a, b)


# ---------------------------------------------------
# region get_backend
# ---------------------------------------------------


def get_backend(name: str, device: DeviceType = "cpu"):
    """Factory: "numpy" | "torch". Explicit, no global state."""
    if name == "numpy":
        return NumpyBackend()
    if name == "torch":
        return TorchBackend(device=device)
    raise ValueError(f"unknown backend: {name!r}")
