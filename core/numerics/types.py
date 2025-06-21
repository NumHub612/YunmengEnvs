# -*- encoding: utf-8 -*-
"""
Copyright (C) 2025, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Type definitions for numerical computations.
"""
import enum


class ElementType(enum.Enum):
    """The element type."""

    CELL = "cell"
    FACE = "face"
    NODE = "node"
    NONE = "none"


class MeshDim(enum.Enum):
    """The dimension of the mesh."""

    DIM1 = "1d"
    DIM2 = "2d"
    DIM3 = "3d"
    NONE = "none"
