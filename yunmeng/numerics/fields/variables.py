# -*- encoding: utf-8 -*-
"""
Copyright (C) 2024, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Variables definition.
"""

from yunmeng.numerics.enums import VariableType
from yunmeng.numerics.fields.backends import (
    Backend,
    backend_context,
    ArrayLike,
    get_backend,
)

import numpy as np
import torch
from typing import Any, Union, Optional, Tuple, Callable

# --------------------------------------------------
# region Variable
# --------------------------------------------------


class Variable:
    """轻量级变量类型视图，不包装运算，仅作为类型视图和操作入口"""

    __slots__ = ("_data", "_type", "_back")

    def __init__(
        self,
        data: ArrayLike,
        vtype: VariableType,
        back: Backend = None,
    ):
        if not vtype.check_shape(data):
            raise ValueError(f"Shape {data.shape} doesn't match type {vtype.name}")

        # Canonicalize scalar storage to 0-d to matches scalar VariableType.shape.
        if vtype.is_scalar and data.shape != ():
            if data.shape == (1,):
                data = data.reshape(())
            else:
                raise ValueError(
                    f"Scalar Variable must have shape () or (1,), got {data.shape}"
                )

        self._data = data
        self._type = vtype
        self._back = back or backend_context.active_backend

    @staticmethod
    def scalar(x: float, backend: Backend = None) -> "Variable":
        back = backend or get_backend()
        data = back.array(x, dtype=back.float64)
        return Variable(data, VariableType.scalar(), back)

    @staticmethod
    def vector(
        x: float, y: float, z: float = 0.0, backend: Backend = None, dim: int = 3
    ) -> "Variable":
        """Create a vector variable.

        Args:
            x: x component
            y: y component
            z: z component (only used for 3D)
            backend: computation backend
            dim: dimension of the vector
        """
        back = backend or get_backend()
        if dim == 2:
            data = back.array([x, y], dtype=back.float64)
        else:
            data = back.array([x, y, z], dtype=back.float64)
        return Variable(data, VariableType.vector(dim), back)

    @staticmethod
    def tensor(
        components: Tuple[float, ...], backend: Backend = None, dim: int = 3
    ) -> "Variable":
        """Create a tensor variable from flat components.

        Args:
            components: dim*dim values for dim*dim tensor
            backend: computation backend
            dim: dimension of the tensor (2 or 3)
        Note:
            components are ordered as (ux, ux, vx, vy) for 2D tensor,
            (ux, uy, uz, vx, vy, vz, wx, wy, wz) for 3D tensor.
        """
        back = backend or get_backend()
        data = back.array(components, dtype=back.float64)
        expected_components = dim * dim
        if len(components) != expected_components:
            raise ValueError(
                f"Expected {expected_components} components for {dim}D tensor, got {len(components)}"
            )
        data = data.reshape((dim, dim))
        return Variable(data, VariableType.tensor(dim), back)

    @staticmethod
    def zeros(vtype: VariableType, backend: Backend = None) -> "Variable":
        back = backend or get_backend()
        shape = (vtype.ncom,)
        data = back.zeros(shape, dtype=back.float64).reshape(vtype.shape)
        return Variable(data, vtype, back)

    @staticmethod
    def from_array(data: ArrayLike, vtype: VariableType = None) -> "Variable":
        """Infer or assign VariableType from array shape."""
        if vtype is None:
            vtype = VariableType.from_shape(data.shape)
        return Variable(data, vtype)

    def to_numpy(self) -> np.ndarray:
        return self._back.to_numpy(self._data)

    def to_tensor(self, device=None, requires_grad: bool = False) -> torch.Tensor:
        t = self._back.to_tensor(self._data, requires_grad=requires_grad)
        if device is not None:
            t = t.to(device)
        return t

    def to_device(self, device):
        return self._back.to_device(self._data, device)

    # -----------------------------------------------
    # region properties
    # -----------------------------------------------

    @property
    def data(self):
        """Data of the variable."""
        return self._data

    @property
    def vtype(self) -> VariableType:
        """Type of the variable."""
        return self._type

    @property
    def dtype(self):
        """Data type."""
        return self._data.dtype

    @property
    def shape(self):
        """Shape of the variable."""
        return self._type.shape

    @property
    def ndim(self):
        """Number of dimensions."""
        return self._type.ndim

    @property
    def ncom(self):
        """Number of components."""
        return self._type.ncom

    # -----------------------------------------------
    # region operators
    # -----------------------------------------------

    def __str__(self) -> str:
        return f"{self._type.name}({self._data})"

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
            if self.vtype.is_scalar:
                # scalar multiplication
                return Variable(
                    self._back.xp.multiply(self._data, other._data),
                    other._type,
                    self._back,
                )
            if self.vtype.is_tensor and other.vtype.is_vector:
                # matrix-vector multiplication
                return Variable(
                    self._back.xp.matmul(self._data, other._data),
                    other.vtype,
                    self._back,
                )
            if self.vtype.is_tensor and other.vtype.is_tensor:
                # matrix-matrix multiplication
                return Variable(
                    self._back.xp.matmul(self._data, other._data),
                    self.vtype,
                    self._back,
                )
            if self.vtype.is_vector and other.vtype.is_tensor:
                # vector-matrix multiplication
                return Variable(
                    self._back.xp.matmul(self._data, other._data),
                    self.vtype,
                    self._back,
                )
            if self.vtype == other.vtype:
                if self.vtype.is_vector:
                    # dot product
                    return Variable.scalar(
                        self._back.xp.dot(self._data, other._data),
                    )
                elif self.vtype.is_scalar:
                    # element-wise multiplication
                    return Variable(
                        self._back.xp.multiply(self._data, other._data),
                        self._type,
                        self._back,
                    )
            raise TypeError("Not supported multiplication.")
        if np.isscalar(other):
            return Variable(
                self._back.xp.multiply(self._data, other),
                self._type,
                self._back,
            )
        return NotImplemented

    __rmul__ = __mul__

    def __truediv__(self, other):
        if isinstance(other, Variable):
            if self.vtype != other.vtype:
                raise TypeError("Not same type.")
            return Variable(
                self._back.xp.divide(self._data, other._data),
                self._type,
                self._back,
            )
        if np.isscalar(other):
            return Variable(
                self._back.xp.divide(self._data, other),
                self._type,
                self._back,
            )
        return NotImplemented

    def __neg__(self):
        return Variable(
            self._back.xp.negative(self._data),
            self._type,
            self._back,
        )


# --------------------------------------------------
# region Var factory
# --------------------------------------------------


def Var(
    value: Union[float, list, tuple, np.ndarray, torch.Tensor, Variable],
    *,
    vtype: VariableType = None,
    backend: Backend = None,
) -> Variable:
    """Factory to create a Variable from various input types.

    REFACTORED: Now supports explicit vtype specification and returns
    a lightweight Variable view.

    Examples:
        >>> Var(1.0)                   # scalar
        >>> Var([1.0, 2.0, 3.0])       # vector (inferred from length 3)
        >>> Var(data, vtype=VariableType.VECTOR)  # explicit type
    """
    if isinstance(value, Variable):
        return value

    back = backend or get_backend()

    if isinstance(value, torch.Tensor):
        data = value
        if vtype is None:
            vtype = VariableType.from_shape(data.shape)
    elif isinstance(value, np.ndarray):
        data = value
        if vtype is None:
            vtype = VariableType.from_shape(data.shape)
    elif isinstance(value, (list, tuple)):
        data = back.array(value)
        if vtype is None:
            vtype = VariableType.from_shape(data.shape)
    elif isinstance(value, (int, float)):
        return Variable.scalar(float(value), back)
    else:
        raise TypeError(f"Cannot create Variable from {type(value)}")

    return Variable(data, vtype, back)
