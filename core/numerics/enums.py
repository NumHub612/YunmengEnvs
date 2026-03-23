# -*- encoding: utf-8 -*-
"""
Copyright (C) 2026, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Enumerations for numerical algorithms and data structures.
"""
from enum import Enum

# --------------------------------------------------
# region Backends
# --------------------------------------------------


class BackendType(Enum):
    """Backend for numerical computation."""

    NUMPY = "numpy"
    TORCH = "torch"


# --------------------------------------------------
# region Mats
# --------------------------------------------------


class EngineMethod(Enum):
    """Engine method for solving linear equations."""

    NUMPY = "numpy"
    SCIPY = "scipy"
    TORCH = "torch"


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
    NONE = "none"  # might be useful for id-based elements


# --------------------------------------------------
# region Fields
# --------------------------------------------------


class VariableType(Enum):
    """Variable types."""

    SCALAR = (1,)
    VECTOR = (3,)
    TENSOR = (3, 3)

    @staticmethod
    def from_shape(value: tuple) -> "VariableType":
        """Get the variable type from the shape."""
        if len(value) == 1:
            return VariableType.SCALAR
        if len(value) == 3 and value[1] == 3 and value[2] == 3:
            return VariableType.TENSOR
        if len(value) == 3:
            return VariableType.VECTOR
        raise ValueError(f"Invalid shape: {value}")

    def check_shape(self, arr) -> bool:
        """Check if the array shape matches the variable type."""
        return arr.shape == self.value
