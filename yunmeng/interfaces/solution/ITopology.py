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
    def layer_id(self) -> str: ...

    @property
    @abstractmethod
    def parent_id(self) -> str | None:
        """ID of the parent layer; None for root layer."""
        ...

    @property
    @abstractmethod
    def spatial_dim(self) -> MeshDimension: ...

    @property
    @abstractmethod
    def grid_type(self) -> str: ...

    @abstractmethod
    def get_nodes(self) -> list[dict]: ...

    @abstractmethod
    def get_edges(self) -> list[dict]: ...

    @abstractmethod
    def get_elements(self) -> list[dict]: ...

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
