# -*- encoding: utf-8 -*-
"""
Copyright (C) 2024, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Variables definition.
"""
from core.numerics.fields.backends import Backend, use_numpy, use_torch, _BACKEND
from configs.settings import settings
import numpy as np
from enum import Enum
from typing import Optional, Any


def Var(arr: float | list | np.ndarray | Any):
    """To create a variable."""
    if isinstance(arr, Variable):
        return arr
    if isinstance(arr, np.ndarray):
        return Variable.from_numpy(arr)
    if isinstance(arr, list):
        return Variable.from_numpy(np.array(arr))
    if isinstance(arr, float):
        return Variable.scalar(arr)
    try:
        import torch

        if isinstance(arr, torch.Tensor):
            return Variable.from_numpy(arr.detach().numpy())
    except ImportError:
        pass
    raise TypeError("Invalid value.")


class VariableType(Enum):
    """Variable types in CFD."""

    SCALAR = (1,)
    VECTOR = (3,)
    TENSOR = (3, 3)

    def check_shape(self, arr) -> bool:
        return arr.shape == self.value

    def from_str(s: str) -> "VariableType":
        if s == "scalar":
            return VariableType.SCALAR
        if s == "vector":
            return VariableType.VECTOR
        if s == "tensor":
            return VariableType.TENSOR
        raise ValueError(f"Invalid variable type: {s}")


class Variable:
    """Variable in CFD."""

    __slots__ = ("_data", "_type", "_back")

    # -----------------------------------------------
    # region constructor
    # -----------------------------------------------

    def __init__(self, data, vtype: VariableType, back: Optional[Backend] = None):
        back = back or _BACKEND
        if not vtype.check_shape(data):
            raise ValueError(f"Shape {data.shape} doesn't match type {vtype.name}")
        self._data = data
        self._type = vtype
        self._back = back

    @staticmethod
    def scalar(x: float, requires_grad: bool = False) -> "Variable":
        """Scalar variable."""
        back = _BACKEND
        data = back.array([x], dtype=back.xp.float64)
        if back.name == "torch":
            data = back.as_tensor(data, requires_grad=requires_grad)
        return Variable(data, VariableType.SCALAR)

    @staticmethod
    def vector(x: float, y: float, z: float, requires_grad=False) -> "Variable":
        """Vector variable."""
        back = _BACKEND
        data = back.array([x, y, z], dtype=back.xp.float64)
        if back.name == "torch":
            data = back.as_tensor(data, requires_grad=requires_grad)
        return Variable(data, VariableType.VECTOR)

    @staticmethod
    def tensor(*args, requires_grad: bool = False) -> "Variable":
        """Tensor variable."""
        back = _BACKEND
        data = back.array(args, dtype=back.xp.float64).reshape(3, 3)
        if back.name == "torch":
            data = back.as_tensor(data, requires_grad=requires_grad)
        return Variable(data, VariableType.TENSOR)

    @staticmethod
    def from_numpy(arr: np.ndarray) -> "Variable":
        """Create a variable from numpy array."""
        if arr.shape == (1,):
            vtype = VariableType.SCALAR
        elif arr.shape == (3,):
            vtype = VariableType.VECTOR
        elif arr.shape == (3, 3):
            vtype = VariableType.TENSOR
        else:
            raise ValueError("Invalid numpy shape.")

        xp = _BACKEND.xp
        data = Backend.from_numpy(arr, xp)
        return Variable(data, vtype, _BACKEND)

    def to_numpy(self) -> np.ndarray:
        return self._back.as_numpy(self._data)

    def zero(self) -> "Variable":
        """Zero variable."""
        return Variable(self._back.zeros_like(self._data), self._type, self._back)

    def to(self, device=settings.device) -> "Variable":
        """To device."""
        _data = self._back.to_device(self._data, device)
        return Variable(_data, self._type, self._back)

    # -----------------------------------------------
    # region properties
    # -----------------------------------------------

    @property
    def data(self):
        """Data of the variable."""
        if self._type == VariableType.SCALAR:
            return self._data[0]
        return self._data

    @property
    def type(self) -> VariableType:
        """Type of the variable."""
        return self._type

    @property
    def shape(self):
        """Shape of the variable."""
        return self._data.shape

    @property
    def magnitude(self) -> float:
        """Magnitude of variable."""
        return self._back.norm(self._data)

    # -----------------------------------------------
    # region operators
    # -----------------------------------------------

    def __str__(self) -> str:
        return f"{self._type.name}({self._back.name}, {self._data})"

    __repr__ = __str__

    def __eq__(self, other: "Variable") -> bool:
        return (
            isinstance(other, Variable)
            and self._type == other._type
            and self._back.xp.allclose(self._data, other._data)
        )

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        scalars = []
        for inp in inputs:
            scalars.append(inp._data if isinstance(inp, Variable) else inp)
        out_raw = getattr(ufunc, method)(*scalars, **kwargs)
        return Variable(out_raw, self._type, self._back)

    def __array_function__(self, func, types, args, kwargs):
        if len(args) == 1 and isinstance(args[0], Variable):
            raw = func(args[0]._data, **kwargs)
            if np.isscalar(raw) or (hasattr(raw, "ndim") and raw.ndim == 0):
                return Variable.scalar(float(raw))
            return Variable(raw, args[0]._type, args[0]._back)
        return NotImplemented

    def __add__(self, other):
        if isinstance(other, Variable):
            if self._type != other._type:
                raise TypeError("Not same type.")
            return Variable(
                self._back.xp.add(self._data, other._data),
                self._type,
                self._back,
            )

        if np.isscalar(other):
            return Variable(
                self._back.xp.add(self._data, other), self._type, self._back
            )
        return NotImplemented

    __radd__ = __add__

    def __sub__(self, other):
        if isinstance(other, Variable):
            if self._type != other._type:
                raise TypeError("Not same type.")
            return Variable(
                self._back.xp.subtract(self._data, other._data),
                self._type,
                self._back,
            )
        if np.isscalar(other):
            return Variable(
                self._back.xp.subtract(self._data, other),
                self._type,
                self._back,
            )
        return NotImplemented

    def __rsub__(self, other):
        if np.isscalar(other):
            return Variable(
                self._back.xp.subtract(other, self._data),
                self._type,
                self._back,
            )
        return NotImplemented

    def __mul__(self, other):
        if isinstance(other, Variable):
            if self._type == VariableType.SCALAR:
                # scalar multiplication
                return Variable(
                    self._back.xp.multiply(self._data, other._data),
                    other._type,
                    self._back,
                )
            if self._type == VariableType.TENSOR and other._type == VariableType.VECTOR:
                # matrix-vector multiplication
                return Variable(
                    self._back.xp.matmul(self._data, other._data),
                    VariableType.VECTOR,
                    self._back,
                )
            if self._type == VariableType.TENSOR and other._type == VariableType.TENSOR:
                # matrix-matrix multiplication
                return Variable(
                    self._back.xp.matmul(self._data, other._data),
                    VariableType.TENSOR,
                    self._back,
                )
            if self._type == other._type:
                if self._type == VariableType.VECTOR:
                    # dot product
                    return Variable.scalar(self._back.xp.dot(self._data, other._data))
                # element-wise multiplication
                return Variable(
                    self._back.xp.multiply(self._data, other._data),
                    self._type,
                    self._back,
                )
            raise TypeError("Not supported multiplication.")
        if np.isscalar(other):
            return Variable(
                self._back.xp.multiply(self._data, other), self._type, self._back
            )
        return NotImplemented

    __rmul__ = __mul__

    def __truediv__(self, other):
        if isinstance(other, Variable):
            if self._type != other._type:
                raise TypeError("Not same type.")
            return Variable(
                self._back.xp.divide(self._data, other._data),
                self._type,
                self._back,
            )
        if np.isscalar(other):
            return Variable(
                self._back.xp.divide(self._data, other), self._type, self._back
            )
        return NotImplemented

    def __neg__(self):
        return Variable(self._back.xp.negative(self._data), self._type, self._back)
