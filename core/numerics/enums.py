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

    @staticmethod
    def from_str(value: str) -> "BackendType":
        try:
            return BackendType(value.upper())
        except ValueError:
            raise ValueError(f"Invalid backend: {value}")


# --------------------------------------------------
# region Mats
# --------------------------------------------------


class EngineMethod(Enum):
    """Engine method for solving linear equations."""

    NUMPY = "numpy"
    SCIPY = "scipy"
    TORCH = "torch"

    @staticmethod
    def from_str(value: str) -> "EngineMethod":
        try:
            return EngineMethod(value.upper())
        except ValueError:
            raise ValueError(f"Invalid engine method: {value}")


# --------------------------------------------------
# region Mesh
# --------------------------------------------------


class MeshDimension(Enum):
    """The mesh dimensions."""

    D1 = "1d"
    D2 = "2d"
    D3 = "3d"
    NONE = "none"

    @staticmethod
    def from_str(value: str) -> "MeshDimension":
        try:
            return MeshDimension(value.upper())
        except ValueError:
            raise ValueError(f"Invalid mesh dimension: {value}")


class GeomType(Enum):
    """The geometry types."""

    IdBased = 0
    Point = 1
    Polyline = 2
    Polygon = 3
    Polyhedron = 4

    @staticmethod
    def from_str(value: str | int) -> "GeomType":
        if isinstance(value, str):
            value = value.upper()
        try:
            return GeomType(value)
        except ValueError:
            raise ValueError(f"Invalid geometry type: {value}")


class ElementType(Enum):
    """Element types in CFD."""

    CELL = "cell"
    FACE = "face"
    NODE = "node"
    NONE = "none"  # might be useful for id-based elements

    @staticmethod
    def from_str(value: str) -> "ElementType":
        try:
            return ElementType(value.upper())
        except ValueError:
            raise ValueError(f"Invalid element type: {value}")


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

    def from_str(name: str) -> "VariableType":
        s = name.lower()
        if s == "scalar":
            return VariableType.SCALAR
        if s == "vector":
            return VariableType.VECTOR
        if s == "tensor":
            return VariableType.TENSOR
        raise ValueError(f"Invalid variable type: {name}")

    def check_shape(self, arr) -> bool:
        """Check if the array shape matches the variable type."""
        return arr.shape == self.value
