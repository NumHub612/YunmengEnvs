# -*- encoding: utf-8 -*-
"""
Copyright (C) 2026, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Backend of variabls and fields.
"""
from configs.settings import settings
import numpy as np
from typing import Optional


class Backend:
    """The backend of the variable."""

    __slots__ = ("xp", "name")

    def __init__(self, xp, name: str):
        self.xp = xp  # np or torch backend
        self.name = name

    @staticmethod
    def from_numpy(arr: np.ndarray, xp):
        if xp.__name__ == "torch":
            import torch

            return torch.from_numpy(arr)
        return arr

    def array(self, obj, dtype=None):
        if self.name == "torch":
            import torch

            return torch.tensor(obj, dtype=dtype)
        return self.xp.array(obj, dtype=dtype)

    def zeros_like(self, arr):
        return self.xp.zeros_like(arr)

    def eye(self, n, dtype):
        return self.xp.eye(n, dtype=dtype)

    def norm(self, arr) -> float:
        if self.name == "torch":
            import torch

            return torch.linalg.norm(arr)
        return float(self.xp.linalg.norm(arr))

    def dot(self, a, b):
        return self.xp.dot(a, b)

    def as_numpy(self, arr) -> np.ndarray:
        if self.name == "torch":
            return arr.detach().cpu().numpy()
        return arr

    def as_tensor(self, obj, dtype=None, requires_grad=False):
        if self.name != "torch":
            raise RuntimeError("Backend isn't torch")
        import torch

        if isinstance(obj, torch.Tensor):
            new_tensor = obj.clone().detach()
            new_tensor.requires_grad_(requires_grad)
            return new_tensor

        return torch.tensor(
            obj,
            dtype=dtype,
            requires_grad=requires_grad,
        )

    def to_device(self, arr, device):
        if self.name == "torch":
            import torch

            return arr.to(device=torch.device(device))
        return arr


_numpy_back = Backend(np, "numpy")
_torch_back: Optional[Backend] = None
_BACKEND: Backend = _numpy_back


def use_numpy():
    global _BACKEND
    _BACKEND = _numpy_back


def use_torch(device=settings.device):
    global _torch_back, _BACKEND
    if _torch_back is None:
        try:
            import torch

            _torch_back = Backend(torch, "torch")
        except ImportError as e:
            raise
    else:
        _torch_back.device = device
    _BACKEND = _torch_back
