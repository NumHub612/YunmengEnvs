# -*- encoding: utf-8 -*-
"""
Copyright (C) 2026, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Enumerations for numerical algorithms and data structures.
"""

from dataclasses import dataclass
from typing import Tuple, Union
from enum import Enum, auto

# ---------------------------------------------------------------------------
# region DeviceType
# ---------------------------------------------------------------------------


class DeviceType(Enum):
    """Compute device type for explicit device placement."""

    CPU = "cpu"
    CUDA = "cuda"
    AUTO = "auto"


# --------------------------------------------------
# region Backends
# --------------------------------------------------


class BackendType(Enum):
    """Backend for numerical computation."""

    NUMPY = "numpy"
    TORCH = "torch"


# --------------------------------------------------
# region VariableType
# --------------------------------------------------


@dataclass(frozen=True)
class VariableType:
    """Variable types with dimension support.

    Type category is determined by ``ndim`` (component rank):
        ndim == 0  ->  scalar   (shape = ())
        ndim == 1  ->  vector   (shape = (dim,))
        ndim == 2  ->  tensor   (shape = (dim, dim))

    Use the `is_scalar`, `is_vector`, `is_tensor` properties
    for type checks instead of string comparisons.
    """

    name: str
    shape: Tuple[int, ...]
    ndim: int
    ncom: int

    @property
    def is_scalar(self) -> bool:
        """True if this is a scalar type (ndim == 0)."""
        return self.ndim == 0

    @property
    def is_vector(self) -> bool:
        """True if this is a vector type (ndim == 1)."""
        return self.ndim == 1

    @property
    def is_tensor(self) -> bool:
        """True if this is a tensor type (ndim == 2)."""
        return self.ndim == 2

    @classmethod
    def scalar(cls) -> "VariableType":
        """Create a scalar variable type."""
        return cls("SCALAR", (), 0, 1)

    @classmethod
    def vector(cls, dim: int) -> "VariableType":
        """Create a vector variable type."""
        if dim < 2:
            raise ValueError(f"Vector dimension must be >= 2, got {dim}")
        return cls("VECTOR", (dim,), 1, dim)

    @classmethod
    def tensor(cls, dim: int) -> "VariableType":
        """Create a tensor variable type."""
        if dim < 2:
            raise ValueError(f"Tensor dimension must be >= 2, got {dim}")
        return cls("TENSOR", (dim, dim), 2, dim * dim)

    def check_shape(self, arr) -> bool:
        """Check if array's trailing dimension match vtype's shape.

        For a field of N elements, a VECTOR field has shape (N, dim),
        so trailing dims (dim,) must match self.shape (dim,).
        """
        if arr is None:
            return False
        trailing = arr.shape[-self.ndim :] if self.ndim > 0 else ()
        return trailing == self.shape

    @classmethod
    def from_shape(cls, shape: Tuple[int, ...]) -> "VariableType":
        """Infer variable type from trailing component dimensions."""
        if len(shape) >= 2 and shape[-2] == shape[-1] and shape[-2] > 0:
            return cls.tensor(shape[-1])
        if len(shape) >= 1 and shape[-1] > 0:
            if shape[-1] == 1:
                return cls.scalar()
            return cls.vector(shape[-1])
        if len(shape) == 0:
            return cls.scalar()
        raise ValueError(
            f"Cannot infer VariableType from shape {shape}; "
            f"supported shapes are scalar, vector(dim), tensor(dim)"
        )

    def __str__(self) -> str:
        """String representation of the variable type."""
        if self.is_scalar:
            return "SCALAR"
        return f"{self.name}({self.shape[-1]})"


# --------------------------------------------------
# region Fields
# --------------------------------------------------


# --------------------------------------------------
# region Mesh
# --------------------------------------------------


class MeshDimension(Enum):
    """The mesh dimensions."""

    NONE = 0
    D1 = 1
    D2 = 2
    D3 = 3


class ElementType(Enum):
    """Element types in CFD."""

    CELL = "cell"
    FACE = "face"
    NODE = "node"
    NONE = "none"
