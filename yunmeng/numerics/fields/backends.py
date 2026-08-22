# -*- encoding: utf-8 -*-
"""
Copyright (C) 2026, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Backend of variables and fields.
"""

from yunmeng.numerics.enums import BackendType, DeviceType
from yunmeng.setting import settings

from typing import Dict, Union
from contextlib import contextmanager
import numpy as np
import torch

# --------------------------------------------------
# region Backend
# --------------------------------------------------

# Backend-neutral device annotation. Runtime values are backend-native
# handles (torch.device under the torch backend); neutral layers must not
# declare concrete torch/np types — conversion happens inside Backend.
DeviceLike = Union[DeviceType, "torch.device", str, None]

ArrayLike = Union[np.ndarray, torch.Tensor]


class Backend:
    """Backend to support torch and numpy."""

    __slots__ = ("xp", "btype")

    def __init__(self, xp, backend_type: BackendType = BackendType.NUMPY):
        assert xp is not None, "xp must be numpy or torch"
        self.xp = xp  # numpy or torch backend
        self.btype = backend_type

    @property
    def is_torch(self) -> bool:
        return self.btype == BackendType.TORCH

    @property
    def is_numpy(self) -> bool:
        return self.btype == BackendType.NUMPY

    @property
    def type(self) -> BackendType:
        return self.btype

    @property
    def float32(self):
        return self.xp.float32

    @property
    def float64(self):
        return self.xp.float64

    def array(self, value, dtype=None, device=None, requires_grad=False) -> ArrayLike:
        dtype = dtype or self.float64
        if self.type == BackendType.TORCH:
            return torch.as_tensor(value, dtype=dtype, device=device).requires_grad_(
                requires_grad
            )
        return self.xp.array(value, dtype=dtype)

    # (FIX) zeros: remove undefined self._device, accept device param instead
    def zeros(self, shape, dtype=None, device=None) -> ArrayLike:
        """Create zero-initialized array."""
        dtype = dtype or self.float64
        if self.is_torch:
            return torch.zeros(shape, dtype=dtype, device=device)
        return np.zeros(shape, dtype=dtype)

    def zeros_like(self, arr) -> ArrayLike:
        return self.xp.zeros_like(arr)

    def empty(self, shape, dtype=None, device=None) -> ArrayLike:
        dtype = dtype or self.float64
        if self.type == BackendType.TORCH:
            return torch.empty(
                shape,
                device=device,
                dtype=dtype,
            )
        else:
            return np.empty(shape, dtype=dtype)

    def eye(self, n, dtype=None) -> ArrayLike:
        dtype = dtype or self.float64
        return self.xp.eye(n, dtype=dtype)

    def full(
        self, shape, fill_value, dtype=None, device=None, requires_grad=False
    ) -> ArrayLike:
        dtype = dtype or self.float64
        if self.type == BackendType.TORCH:
            data = torch.full(
                shape,
                fill_value,
                dtype=dtype,
                device=device,
                requires_grad=requires_grad,
            )
        else:
            data = np.full(
                shape,
                fill_value,
                dtype=dtype,
            )
        return data

    def stack(self, arr, axis=0) -> ArrayLike:
        if self.type == BackendType.TORCH:
            device0 = arr[0].device
            arr = [t.to(device0) for t in arr]
            return torch.stack(arr, dim=axis)
        return np.stack(arr, axis=axis)

    # --------------------------------------------------
    # region ArrayNamespace primitives
    # --------------------------------------------------
    # Operator/Field layers must dispatch through these
    # primitives instead of touching np./torch. directly.

    def asarray(self, obj, dtype=None, device=None) -> ArrayLike:
        """Convert array-like to backend array without copying when possible.

        Note: converting across backends (e.g. torch -> numpy) detaches the
        autograd graph by nature. Intra-backend conversion keeps the graph.
        """
        if self.is_torch:
            if isinstance(obj, torch.Tensor):
                t = obj
                if dtype is not None:
                    t = t.to(dtype)
                if device is not None:
                    t = t.to(device)
                return t
            return torch.as_tensor(obj, dtype=dtype, device=device)
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().numpy()
        return np.asarray(obj, dtype=dtype)

    def where(self, cond, x, y) -> ArrayLike:
        """Element-wise selection. Differentiable in torch (subgradient at
        the switching surface is zero, acceptable for upwind/limiter logic)."""
        return self.xp.where(cond, x, y)

    def maximum(self, a, b) -> ArrayLike:
        if self.is_torch:
            return torch.maximum(self.asarray(a), self.asarray(b))
        return np.maximum(a, b)

    def minimum(self, a, b) -> ArrayLike:
        if self.is_torch:
            return torch.minimum(self.asarray(a), self.asarray(b))
        return np.minimum(a, b)

    def ones_like(self, arr) -> ArrayLike:
        return self.xp.ones_like(arr)

    def full_like(self, arr, fill_value) -> ArrayLike:
        if self.is_torch:
            return torch.full_like(arr, fill_value)
        return np.full_like(arr, fill_value)

    def arange(self, *args, dtype=None, device=None) -> ArrayLike:
        if self.is_torch:
            return torch.arange(*args, dtype=dtype, device=device)
        return np.arange(*args, dtype=dtype)

    def concatenate(self, arrs, axis=0) -> ArrayLike:
        if self.is_torch:
            return torch.cat(list(arrs), dim=axis)
        return np.concatenate(list(arrs), axis=axis)

    def sum(self, arr, axis=None, keepdims=False):
        if self.is_torch:
            return torch.sum(arr, dim=axis, keepdim=keepdims)
        return np.sum(arr, axis=axis, keepdims=keepdims)

    def mean(self, arr, axis=None, keepdims=False):
        if self.is_torch:
            return torch.mean(arr, dim=axis, keepdim=keepdims)
        return np.mean(arr, axis=axis, keepdims=keepdims)

    def sqrt(self, arr) -> ArrayLike:
        return self.xp.sqrt(arr)

    def is_tensor(self, obj) -> bool:
        """Whether obj is the native array type of this backend."""
        if self.is_torch:
            return isinstance(obj, torch.Tensor)
        return isinstance(obj, np.ndarray)

    # --------------------------------------------------
    # region Algorithmic primitives
    # --------------------------------------------------
    # Beyond data-type abstraction, the backend also abstracts ALGORITHMIC
    # primitives whose optimal implementations diverge across backends
    # (sparse ops, scatter/gather, linear solves). Operator code must call
    # these instead of hand-rolling np./torch. variants. Heavy paths may
    # have per-backend optimized implementations behind the same signature.

    def matmul(self, a, b) -> ArrayLike:
        if self.is_torch:
            return torch.matmul(a, b)
        return np.matmul(a, b)

    def einsum(self, equation, *operands) -> ArrayLike:
        if self.is_torch:
            return torch.einsum(equation, *operands)
        return np.einsum(equation, *operands)

    def scatter_add(self, index, src, dim_size: int) -> ArrayLike:
        """Segment/scatter accumulation — the core primitive of FEM/FVM
        assembly and mesh-based GNN message passing.

        Numpy: np.add.at (slow but correct). Torch: index_add_ (fast, on
        device, differentiable)."""
        if self.is_torch:
            if not torch.is_tensor(index):
                index = torch.as_tensor(index, dtype=torch.int64, device=src.device)
            out = torch.zeros(
                (dim_size, *src.shape[1:]), dtype=src.dtype, device=src.device
            )
            out.index_add_(0, index, src)
            return out
        out = np.zeros((dim_size, *src.shape[1:]), dtype=src.dtype)
        np.add.at(out, index, src)
        return out

    def solve(self, a, b) -> ArrayLike:
        """Dense linear solve (differentiable under torch).

        Sparse/iterative solves are NOT covered here: they are handled by
        numerics.linalgs with per-backend implementations; under torch a
        differentiable sparse solve must be wrapped via the implicit
        function theorem (custom autograd.Function), not plain AD through
        the iterations."""
        if self.is_torch:
            return torch.linalg.solve(a, b)
        return np.linalg.solve(a, b)

    def norm(self, arr) -> float:
        if self.type == BackendType.TORCH:
            return torch.linalg.norm(arr)
        return float(self.xp.linalg.norm(arr))

    def dot(self, a, b):
        return self.xp.dot(a, b)

    def abs(self, arr):
        if self.type == BackendType.TORCH:
            return torch.abs(arr)
        return self.xp.abs(arr)

    def min(self, arr):
        if self.type == BackendType.TORCH:
            return torch.min(arr).item()
        return float(self.xp.min(arr))

    def max(self, arr):
        if self.type == BackendType.TORCH:
            return torch.max(arr).item()
        return float(self.xp.max(arr))

    def to_numpy(self, obj) -> np.ndarray:
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().numpy()
        return obj

    def to_tensor(self, obj, dtype=None, requires_grad=False):
        if isinstance(obj, torch.Tensor):
            obj.requires_grad_(requires_grad)
            return obj

        return torch.tensor(
            obj,
            dtype=dtype,
            requires_grad=requires_grad,
        )

    def to_device(self, obj, device=None):
        if self.type == BackendType.TORCH:
            device = device or settings.device
            device = torch.device(device)
            return obj.to(device)
        return obj


# --------------------------------------------------
# region BackendContext
# --------------------------------------------------


class BackendContext:
    """Backend context manager, supports per-context Backend instantiation."""

    _instance: "BackendContext" = None
    _backends: Dict[BackendType, Dict[DeviceType, Backend]] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, "_initialized"):
            self._initialized = True
            self._active_backend: Backend = None
            self._active_device: DeviceType = DeviceType.AUTO

    @contextmanager
    def use_backend(
        self,
        backend_type: BackendType,
        device: DeviceType = DeviceType.AUTO,
    ):
        """A context manager that uses the specified backend and device."""
        prev_backend = self._active_backend
        prev_device = self._active_device

        try:
            if backend_type not in self._backends:
                self._backends[backend_type] = {}

            if device not in self._backends[backend_type]:
                xp = np if backend_type == BackendType.NUMPY else torch
                self._backends[backend_type][device] = Backend(xp, backend_type)

            self._active_backend = self._backends[backend_type][device]
            self._active_device = device
            yield self._active_backend
        finally:
            self._active_backend = prev_backend
            self._active_device = prev_device

    @property
    def active_backend(self) -> Backend:
        """Get the current active backend."""
        if self._active_backend is None:
            raise RuntimeError(
                "No backend is active. Use 'use_backend' context manager."
            )
        return self._active_backend

    @property
    def active_device(self) -> DeviceType:
        """Get the current active device."""
        return self._active_device


# Global backend context instance
backend_context = BackendContext()

# --------------------------------------------------
# region Conveniences
# --------------------------------------------------

__numpy_back = Backend(np, BackendType.NUMPY)
__torch_back = Backend(torch, BackendType.TORCH)


def use_numpy():
    """Get numpy backend."""
    return __numpy_back


def use_torch():
    """Get torch backend."""
    return __torch_back


def get_backend(type: BackendType = None):
    """Get backend. Priority: active context > explicit type > settings default."""
    # 1) If a context is active and type matches (or not specified), use it
    try:
        active = backend_context.active_backend
        if type is None or active.type == type:
            return active
    except RuntimeError:
        pass

    # 2) Fallback to explicit type or settings
    if type is None:
        type = BackendType.TORCH if settings.device == "cuda" else BackendType.NUMPY

    if type == BackendType.NUMPY:
        return __numpy_back
    elif type == BackendType.TORCH:
        return __torch_back
    else:
        raise ValueError(f"Unknown backend type: {type}")
