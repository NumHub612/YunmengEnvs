# -*- encoding: utf-8 -*-
"""
Copyright (C) 2024, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Variables definition.
"""
from core.numerics.fields.backends import Backend, get_backend
import numpy as np
import torch
from enum import Enum
from typing import Optional, Any


def Var(arr: float | list | np.ndarray | Any):
    """To create a variable."""
    if isinstance(arr, Variable):
        return arr
    if isinstance(arr, torch.Tensor):
        return Variable.from_numpy(arr.detach().numpy())
    if isinstance(arr, np.ndarray):
        return Variable.from_numpy(arr)
    if isinstance(arr, list):
        return Variable.from_numpy(np.array(arr))
    if isinstance(arr, float):
        return Variable.scalar(arr)

    raise TypeError("Invalid value.")


class VariableType(Enum):
    """Variable types."""

    SCALAR = (1,)
    VECTOR = (3,)
    TENSOR = (3, 3)

    def from_shape(shape: tuple) -> "VariableType":
        if len(shape) == 1:
            return VariableType.SCALAR
        if len(shape) == 3 and shape[1] == 3 and shape[2] == 3:
            return VariableType.TENSOR
        if len(shape) == 3:
            return VariableType.VECTOR
        raise ValueError(f"Invalid shape: {shape}")

    def from_str(s: str) -> "VariableType":
        if s == "scalar":
            return VariableType.SCALAR
        if s == "vector":
            return VariableType.VECTOR
        if s == "tensor":
            return VariableType.TENSOR
        raise ValueError(f"Invalid variable type: {s}")

    def check_shape(self, arr) -> bool:
        return arr.shape == self.value


class Variable:
    """Variable for Scalar, Vector, Tensor."""

    __slots__ = ("_data", "_type", "_back")

    # -----------------------------------------------
    # region constructor
    # -----------------------------------------------

    def __init__(
        self,
        data: np.ndarray | torch.Tensor,
        vtype: VariableType,
        back: Optional[Backend] = None,
    ):
        if not vtype.check_shape(data):
            raise ValueError(f"Shape {data.shape} doesn't match type {vtype.name}")
        self._data = data
        self._type = vtype
        self._back = back or get_backend()

    @staticmethod
    def scalar(x: float, requires_grad: bool = False) -> "Variable":
        """Scalar variable."""
        back = get_backend()
        data = back.array(
            [x],
            dtype=back.xp.float64,
            requires_grad=requires_grad,
        )
        return Variable(data, VariableType.SCALAR)

    @staticmethod
    def vector(x: float, y: float, z: float, requires_grad=False) -> "Variable":
        """Vector variable."""
        back = get_backend()
        data = back.array(
            [x, y, z],
            dtype=back.xp.float64,
            requires_grad=requires_grad,
        )
        return Variable(data, VariableType.VECTOR)

    @staticmethod
    def tensor(*args, requires_grad: bool = False) -> "Variable":
        """Tensor variable."""
        back = get_backend()
        data = back.array(
            args, dtype=back.xp.float64, requires_grad=requires_grad
        ).reshape(3, 3)
        return Variable(data, VariableType.TENSOR)

    @staticmethod
    def from_numpy(arr: np.ndarray) -> "Variable":
        if arr.shape == (1,):
            vtype = VariableType.SCALAR
        elif arr.shape == (3,):
            vtype = VariableType.VECTOR
        elif arr.shape == (3, 3):
            vtype = VariableType.TENSOR
        else:
            raise ValueError("Invalid numpy shape.")

        back = get_backend()
        xp = back.xp
        data = Backend.from_numpy(arr, xp)
        return Variable(data, vtype, back)

    def to_numpy(self) -> np.ndarray:
        return self._back.as_numpy(self._data)

    def zero(self) -> "Variable":
        return Variable(
            self._back.zeros_like(self._data),
            self._type,
            self._back,
        )

    def to(self, device) -> "Variable":
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
    def magnitude(self):
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
        """To support numpy ufuncs, such as np.sin, etc."""
        scalars = []
        for inp in inputs:
            scalars.append(inp._data if isinstance(inp, Variable) else inp)
        out_raw = getattr(ufunc, method)(*scalars, **kwargs)
        return Variable(out_raw, self._type, self._back)

    def __array_function__(self, func, types, args, kwargs):
        """To support numpy functions, such as np.sum, etc."""
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
                self._back.xp.add(self._data, other),
                self._type,
                self._back,
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
            if self.type == VariableType.SCALAR:
                # scalar multiplication
                return Variable(
                    self._back.xp.multiply(self.data, other.data),
                    other.type,
                    self._back,
                )
            if self.type == VariableType.TENSOR and other.type == VariableType.VECTOR:
                # matrix-vector multiplication
                return Variable(
                    self._back.xp.matmul(self.data, other.data),
                    VariableType.VECTOR,
                    self._back,
                )
            if self.type == VariableType.TENSOR and other.type == VariableType.TENSOR:
                # matrix-matrix multiplication
                return Variable(
                    self._back.xp.matmul(self.data, other.data),
                    VariableType.TENSOR,
                    self._back,
                )
            if self.type == other.type:
                if self.type == VariableType.VECTOR:
                    # dot product
                    return Variable.scalar(
                        self._back.xp.dot(self.data, other.data),
                    )
                # element-wise multiplication
                return Variable(
                    self._back.xp.multiply(self.data, other.data),
                    self.type,
                    self._back,
                )
            raise TypeError("Not supported multiplication.")
        if np.isscalar(other):
            return Variable(
                self._back.xp.multiply(self.data, other),
                self.type,
                self._back,
            )
        return NotImplemented

    __rmul__ = __mul__

    def __truediv__(self, other):
        if isinstance(other, Variable):
            if self.type != other.type:
                raise TypeError("Not same type.")
            return Variable(
                self._back.xp.divide(self.data, other.data),
                self.type,
                self._back,
            )
        if np.isscalar(other):
            return Variable(
                self._back.xp.divide(self.data, other),
                self.type,
                self._back,
            )
        return NotImplemented

    def __neg__(self):
        return Variable(
            self._back.xp.negative(self.data),
            self.type,
            self._back,
        )
