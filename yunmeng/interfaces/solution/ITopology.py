# -*- encoding: utf-8 -*-
"""
Copyright (C) 2026, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Dimension-aware internal topology interfaces.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from yunmeng.interfaces.types import ArrayLike, MeshDimension

# ---------------------------------------------------
# region ITopologyLayer
# ---------------------------------------------------


class ITopologyLayer(ABC):
    """Abstraction for a layer of the internal topology."""

    ...


# ---------------------------------------------------
# region ISpatialIndex
# ---------------------------------------------------


class ISpatialIndex(ABC):
    """Fast spatial queries on the internal topology."""

    @abstractmethod
    def find_nearest(
        self,
        x: float,
        y: float,
        z: float = 0.0,
    ) -> str: ...

    @abstractmethod
    def find_containing(
        self,
        x: float,
        y: float,
        z: float = 0.0,
    ) -> str: ...

    @abstractmethod
    def find_within_radius(
        self,
        x: float,
        y: float,
        radius: float,
    ) -> list[str]: ...

    @abstractmethod
    def interpolate_to_point(
        self,
        values: dict[str, ArrayLike],
        x: float,
        y: float,
        z: float = 0.0,
    ) -> dict[str, float]: ...
