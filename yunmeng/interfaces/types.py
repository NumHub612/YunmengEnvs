# -*- encoding: utf-8 -*-
"""
Copyright (C) 2026, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Core value types and aliases for the unified interfaces layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, TypeAlias

# ---------------------------------------------------
# region Backend aliases
# ---------------------------------------------------

ArrayLike: TypeAlias = Any
"""Backend array or array-compatible object (numpy.ndarray,
torch.Tensor, jax.Array, ...). The interfaces layer never names the
concrete type; computations on ArrayLike values must go through the
Backend protocol, never through direct library calls."""

DeviceType: TypeAlias = str
"""Device specifier string, e.g. "cpu", "cuda", "cuda:0".
Open-ended by nature (device ordinals), hence a str alias rather
than an enum."""

Variable: TypeAlias = ArrayLike
"""Alias kept for readability in value-carrying signatures
(boundary values, initial conditions, parameters)."""

# ---------------------------------------------------
# region Env enums
# ---------------------------------------------------


class RunMode(Enum):
    """Solver runtime behavior mode.

    TRAIN: graph-preserving — no detach, cache bypassed, operator
           outputs freshly allocated.
    EVAL:  inference-optimized (default, backward compatible).
    """

    TRAIN = "train"
    EVAL = "eval"


# ---------------------------------------------------
# region Spatial enums
# ---------------------------------------------------


class MeshDimension(Enum):
    """Spatial dimension of a mesh."""

    NONE = "none"
    D1 = "1d"
    D2 = "2d"
    D3 = "3d"


class ElementType(Enum):
    """Mesh element location."""

    CELL = "cell"
    FACE = "face"
    NODE = "node"
    NONE = "none"


class VariableType(Enum):
    """Tensor rank of a variable."""

    SCALAR = "scalar"
    VECTOR = "vector"
    TENSOR = "tensor"


class GeometryType(Enum):
    """Geometry kind of a topology layer."""

    IDBASED = "idbased"
    POINT = "point"
    POLYLINE = "polyline"
    POLYGON = "polygon"
    CUBE = "cube"


# ---------------------------------------------------
# region Shared values
# ---------------------------------------------------


@dataclass
class ParamMeta:
    """Tunable parameter descriptor."""

    name: str
    description: str = ""
    dtype: str = "float"
    bounds: tuple = (None, None)  # (min, max)
    default: Any = None
    required: bool = False
