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


class ElementType(enum.Enum):
    """Element types in CFD."""

    CELL = "cell"
    FACE = "face"
    EDGE = "edge"  # not used.
    NODE = "node"
    NONE = "none"  # might be useful for id-based elements


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
