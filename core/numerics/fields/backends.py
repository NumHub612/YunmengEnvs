# -*- encoding: utf-8 -*-
"""
Copyright (C) 2026, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Backend of variabls and fields.
"""
from configs.settings import settings
import numpy as np
import torch


class Backend:
    """Backend to support torch and numpy."""

    __slots__ = ("xp", "name")

    def __init__(self, xp, name: str):
        assert xp is not None, "xp must be numpy or torch"
        assert name in ("numpy", "torch"), "name must be numpy or torch"
        self.xp = xp  # numpy or torch backend
        self.name = name

    @staticmethod
    def from_numpy(arr: np.ndarray, xp):
        if xp.__name__ == "torch":
            return torch.from_numpy(arr)
        return arr

    def array(self, obj, dtype=None, requires_grad=False):
        if self.name == "torch":
            return torch.tensor(
                obj,
                dtype=dtype,
                requires_grad=requires_grad,
            )
        return self.xp.array(obj, dtype=dtype)

    def zeros_like(self, arr):
        return self.xp.zeros_like(arr)

    def eye(self, n, dtype):
        return self.xp.eye(n, dtype=dtype)

    def norm(self, arr) -> float:
        if self.name == "torch":
            return torch.linalg.norm(arr)
        return float(self.xp.linalg.norm(arr))

    def dot(self, a, b):
        return self.xp.dot(a, b)

    def to_numpy(self, arr) -> np.ndarray:
        if self.name == "torch":
            return arr.detach().cpu().numpy()
        return arr

    def to_tensor(self, obj, dtype=None, requires_grad=False):
        if self.name != "torch":
            raise RuntimeError("Backend isn't torch")

        if isinstance(obj, torch.Tensor):
            _tensor = obj.clone().detach()
            _tensor.requires_grad_(requires_grad)
            return _tensor

        return torch.tensor(
            obj,
            dtype=dtype,
            requires_grad=requires_grad,
        )

    def to_device(self, arr, device=None):
        if self.name == "torch":
            device = device or settings.device
            device = torch.device(device)
            return arr.to(device)
        return arr


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


def get_backend():
    """Get backend according to settings.device."""
    if settings.device == "cuda":
        return __torch_back
    else:
        return __numpy_back
