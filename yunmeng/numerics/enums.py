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


@dataclass
class VariableMeta:
    """Descriptor for variable type metadata."""

    shape: Tuple[int, ...]
    ndim: int
    ncom: int


class VariableType(Enum):
    """Variable types."""

    SCALAR = VariableMeta((), 0, 1)
    VECTOR = VariableMeta((3,), 1, 3)
    TENSOR = VariableMeta((3, 3), 2, 9)

    @property
    def shape(self) -> Tuple[int, ...]:
        """Component shape — e.g. (), (3,), (3,3)."""
        return self.value.shape

    @property
    def ndim(self) -> int:
        """Number of component dimensions."""
        return self.value.ndim

    @property
    def ncom(self) -> int:
        """Total scalar components (e.g. 1, 3, 9)."""
        return self.value.ncom

    def check_shape(self, arr) -> bool:
        """Check if array's trailing dimension match vtype's shape.

        For a field of N elements, a VECTOR field has shape (N, 3),
        so trailing dims (3,) must match self.shape (3,).
        """
        if arr is None:
            return False
        trailing = arr.shape[-self.ndim :] if self.ndim > 0 else ()
        return trailing == self.shape

    @classmethod
    def from_shape(cls, shape: Tuple[int, ...]) -> "VariableType":
        """Infer variable type from trailing component dimensions.

        Works for both single-variable shapes and field shapes:
            ()          -> SCALAR
            (1,)        -> SCALAR
            (3,)        -> VECTOR
            (3, 3)      -> TENSOR
            (N, 3)      -> VECTOR
            (N, 3, 3)   -> TENSOR
        """
        if len(shape) >= 2 and shape[-2:] == (3, 3):
            return cls.TENSOR
        if len(shape) >= 1 and shape[-1:] == (3,):
            return cls.VECTOR
        if len(shape) == 0 or shape[-1:] == (1,):
            return cls.SCALAR
        raise ValueError(
            f"Cannot infer VariableType from shape {shape}; "
            f"supported component shapes are {cls.SCALAR.shape}, "
            f"{cls.VECTOR.shape}, {cls.TENSOR.shape}"
        )


# --------------------------------------------------
# region Fields
# --------------------------------------------------


# --------------------------------------------------
# region Mesh
# --------------------------------------------------


class MeshDimension(Enum):
    """The mesh dimensions."""

    D1 = "1d"
    D2 = "2d"
    D3 = "3d"
    NONE = "none"


class GeomType(Enum):
    """The geometry types."""

    IdBased = 0
    Point = 1
    Polyline = 2
    Polygon = 3
    Polyhedron = 4


class ElementType(Enum):
    """Element types in CFD."""

    CELL = "cell"
    FACE = "face"
    NODE = "node"
    NONE = "none"
