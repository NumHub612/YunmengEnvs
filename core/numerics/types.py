# -*- encoding: utf-8 -*-
"""
Copyright (C) 2025, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Type definitions for numerical computations.
"""
import enum


class VariableType(enum.Enum):
    """Variable types in CFD."""

    SCALAR = "scalar"
    VECTOR = "vector"
    TENSOR = "tensor"
    NONE = "none"

    @staticmethod
    def from_str(vtype: str) -> "VariableType":
        """Convert a string to a VariableType."""
        if vtype == "scalar":
            return VariableType.SCALAR
        elif vtype == "vector":
            return VariableType.VECTOR
        elif vtype == "tensor":
            return VariableType.TENSOR
        else:
            return VariableType.NONE


class ElementType(enum.Enum):
    """Element types in CFD."""

    CELL = "cell"
    FACE = "face"
    EDGE = "edge"  # not used.
    NODE = "node"
    NONE = "none"  # might be useful for id-based elements

    @staticmethod
    def from_str(etype: str) -> "ElementType":
        """Convert a string to an ElementType."""
        if etype == "cell":
            return ElementType.CELL
        elif etype == "face":
            return ElementType.FACE
        elif etype == "edge":
            return ElementType.EDGE
        elif etype == "node":
            return ElementType.NODE
        else:
            return ElementType.NONE


class GeomType(enum.Enum):
    """The geometry types."""

    IdBased = 0
    Point = 1
    Polyline = 2
    Polygon = 3
    Polyhedron = 4


class MeshDim(enum.Enum):
    """The mesh dimensions."""

    DIM1 = "1d"
    DIM2 = "2d"
    DIM3 = "3d"
    NONE = "none"

    @staticmethod
    def from_str(dim: str) -> "MeshDim":
        """Convert a string to a MeshDim."""
        if dim == "1d":
            return MeshDim.DIM1
        elif dim == "2d":
            return MeshDim.DIM2
        elif dim == "3d":
            return MeshDim.DIM3
        else:
            return MeshDim.NONE
