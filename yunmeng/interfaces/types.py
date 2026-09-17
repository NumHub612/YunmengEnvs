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


class MeshDimension(Enum):
    """Spatial dimension of a mesh."""

    D3 = "3d"
    D2 = "2d"
    D1 = "1d"
    NONE = "none"


class ElementType(Enum):
    """Mesh element location."""

    CELL = "cell"
    FACE = "face"
    NODE = "node"
    NONE = "none"


class GeometryType(Enum):
    """Geometry kind of a topology layer."""

    IDBASED = "idbased"
    POINT = "point"
    POLYLINE = "polyline"
    POLYGON = "polygon"
    CUBE = "cube"


class VariableType(Enum):
    """Variable tensor rank descriptor."""

    SCALAR = ()
    VECTOR = (3,)
    TENSOR = (3, 3)

    @property
    def shape(self) -> tuple[int, ...]:
        return self.value

    @property
    def n_dimension(self) -> int:
        return len(self.value)

    @property
    def n_component(self) -> int:
        n = 1
        for d in self.value:
            n *= d
        return n
