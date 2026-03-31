# -*- encoding: utf-8 -*-
"""
Copyright (C) 2026, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Backend of variabls and fields.
"""
from configs.settings import settings
from core.numerics.enums import BackendType
import numpy as np
import torch


class Backend:
    """Backend to support torch and numpy."""

    __slots__ = ("xp", "btype")

    def __init__(self, xp, name: str):
        assert xp is not None, "xp must be numpy or torch"
        assert name in ("numpy", "torch"), "name must be numpy or torch"
        self.xp = xp  # numpy or torch backend
        self.btype = BackendType[name.upper()]

    @property
    def type(self) -> BackendType:
        return self.btype

    @property
    def float32(self):
        return self.xp.float32

    @property
    def float64(self):
        return self.xp.float64

    def data(self, value, dtype=None, gpu=None):
        dtype = dtype or self.float64
        if self.type == BackendType.TORCH:
            return torch.as_tensor(value, dtype=dtype, device=gpu)
        else:
            return np.array(value, dtype=dtype)

    def array(self, obj, dtype=None, requires_grad=False, gpu=None):
        dtype = dtype or self.float64
        if self.type == BackendType.TORCH:
            return torch.as_tensor(obj, dtype=dtype, device=gpu).requires_grad_(
                requires_grad
            )
        return self.xp.array(obj, dtype=dtype)

    def zeros_like(self, arr):
        return self.xp.zeros_like(arr)

    def empty(self, shape, dtype=None, device=None):
        dtype = dtype or self.float64
        if self.type == BackendType.TORCH:
            return torch.empty(
                shape,
                device=device,
                dtype=dtype,
            )
        else:
            return np.empty(shape, dtype=dtype)

    def eye(self, n, dtype=None):
        dtype = dtype or self.float64
        return self.xp.eye(n, dtype=dtype)

    def full(self, shape, fill_value, dtype=None, gpu=None, requires_grad=False):
        dtype = dtype or self.float64
        if self.type == BackendType.TORCH:
            with torch.cuda.device(gpu):
                data = torch.full(
                    shape,
                    fill_value,
                    dtype=dtype,
                    device=gpu,
                    requires_grad=requires_grad,
                )
        else:
            data = np.full(
                shape,
                fill_value,
                dtype=dtype,
            )
        return data

    def stack(self, arr, axis=0):
        if self.type == BackendType.TORCH:
            device0 = arr[0].device
            arr = [t.to(device0) for t in arr]
            return torch.stack(arr, dim=axis)
        return np.stack(arr, axis=axis)

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
            # obj = obj.clone().detach()
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


__numpy_back = Backend(np, "numpy")
__torch_back = Backend(torch, "torch")


def use_numpy():
    """Get numpy backend."""
    return __numpy_back


def use_torch():
    """Get torch backend."""
    global __torch_back
    if __torch_back is None:
        try:
            __torch_back = Backend(torch, "torch")
        except ImportError as e:
            raise
    return __torch_back


def get_backend(type: BackendType = None):
    """Get backend."""
    if type is None:
        type = BackendType.TORCH if settings.device == "cuda" else BackendType.NUMPY

    if type == BackendType.NUMPY:
        return __numpy_back
    elif type == BackendType.TORCH:
        return __torch_back
    else:
        raise ValueError(f"Unknown backend type: {type}")
