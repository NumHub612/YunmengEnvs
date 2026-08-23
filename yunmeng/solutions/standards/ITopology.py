# -*- encoding: utf-8 -*-
"""
Copyright (C) 2025, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Dimension-aware internal topology interfaces.

Design principles:
  1. Hierarchical — a basin contains sub-basins, which contain grid cells
  2. Dimension-aware — 1-D chains, 2-D grids, 3-D volumes are first-class
  3. Queryable — geometry lookups independent of how data is stored
  4. Lazy — expensive structures are built on demand, not at import
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from enum import Enum
import numpy as np
from yunmeng.numerics.enums import MeshDimension

# ---------------------------------------------------
# region Geometry type
# ---------------------------------------------------


class GeometryType(Enum):
    """
    Enum for geometry type.
    """

    IDBASED = "idbased"
    POINT = "point"
    POLYLINE = "polyline"
    POLYGON = "polygon"
    CUBE = "cube"


# ---------------------------------------------------
# region ITopologyLayer
# ---------------------------------------------------


class ITopologyLayer(ABC):
    """A single layer in a potentially hierarchical topology.

    A simple lumped model has one layer (element_count=1).
    A distributed rainfall-runoff model may have:
        Layer 0: basin level
        Layer 1: grid level
    A 1-D/2-D coupled model may have:
        Layer 0: 1-D channel network
        Layer 1: 2-D floodplain grid
    """

    @property
    @abstractmethod
    def layer_id(self) -> str:
        """Unique identifier for this layer."""
        pass

    @property
    @abstractmethod
    def parent_id(self) -> str:
        """ID of the parent layer; None for root layer."""
        pass

    @property
    @abstractmethod
    def spatial_dim(self) -> MeshDimension:
        pass

    @property
    @abstractmethod
    def grid_type(self) -> str:
        pass

    @abstractmethod
    def get_nodes(self) -> list[dict]:
        """Return nodes in this layer."""
        pass

    @abstractmethod
    def get_edges(self) -> list[dict]:
        """Return directed edges in this layer."""
        pass

    @abstractmethod
    def get_elements(self) -> list[dict]:
        """Return compute elements."""
        pass

    def get_parent_map(self) -> dict[str, str]:
        """Return mapping from element id → parent element id
        in the layer above.
        """
        return {}

    def get_children(self, element_id: str) -> list[str]:
        """Return child element ids in the layer below."""
        return []


# ---------------------------------------------------
# region ISpatialIndex
# ---------------------------------------------------


class ISpatialIndex(ABC):
    """Fast spatial queries on the internal topology."""

    @abstractmethod
    def find_nearest(self, x: float, y: float, z: float = 0.0) -> str:
        """Return the id of the nearest element."""
        pass

    @abstractmethod
    def find_containing(
        self,
        x: float,
        y: float,
        z: float = 0.0,
    ) -> str:
        """Return the id of the element that contains the point,
        or None if the point is outside the domain."""
        pass

    @abstractmethod
    def find_within_radius(
        self,
        x: float,
        y: float,
        radius: float,
    ) -> list[str]:
        """Return ids of all elements within *radius*."""
        pass

    @abstractmethod
    def interpolate_to_point(
        self,
        values: dict[str, np.ndarray],
        x: float,
        y: float,
        z: float = 0.0,
    ) -> dict[str, float]:
        """Interpolate field values to an arbitrary point."""
        pass
