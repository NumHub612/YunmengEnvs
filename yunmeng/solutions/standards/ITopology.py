# -*- encoding: utf-8 -*-
"""
Copyright (C) 2025, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Dimension-aware internal topology interfaces.

IInternalTopology exposes a component's internal spatial structure
in a way that supports:

  - 1-D: river reaches with cross-sections, channel networks
  - 2-D: raster grids (DEM-based), unstructured meshes (triangles, quads)
  - 3-D: layered lakes, estuaries, coastal oceans
  - Mixed: 1-D channel + 2-D floodplain, nested basins

Design principles:
  1. Hierarchical — a basin contains sub-basins, which contain grid cells
  2. Dimension-aware — 1-D chains, 2-D grids, 3-D volumes are first-class
  3. Queryable — geometry lookups independent of how data is stored
  4. Lazy — expensive structures are built on demand, not at import
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import numpy as np

from IComponent import ExchangeMeta

# ---------------------------------------------------
# region TopologyMeta
# ---------------------------------------------------


@dataclass
class TopologyMeta:
    """High-level summary of a component's internal topology."""

    spatial_dim: str
    grid_type: str
    element_count: int = 0

    node_count: int = 0
    edge_count: int = 0

    has_hierarchy: bool = False
    """Whether the topology has multiple nested layers
    (e.g. basin → sub-basin → grid cell)."""

    layer_count: int = 1
    """For 3-D or 2-D models: number of vertical layers.
    For pure 2-D: 1.  For pure 1-D: 0 or 1."""

    bounds: tuple = field(default_factory=lambda: (0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
    """Axis-aligned bounding box: (xmin, ymin, zmin, xmax, ymax, zmax).
    For 1-D models ymin==ymax and zmin==zmax."""

    crs: str = ""
    """Coordinate reference system."""

    def to_dict(self) -> dict:
        return {
            "spatial_dim": self.spatial_dim,
            "grid_type": self.grid_type,
            "element_count": self.element_count,
            "node_count": self.node_count,
            "edge_count": self.edge_count,
            "has_hierarchy": self.has_hierarchy,
            "layer_count": self.layer_count,
            "bounds": self.bounds,
            "crs": self.crs,
        }


# ---------------------------------------------------------------------------
# region IInternalTopology
# ---------------------------------------------------------------------------


class IInternalTopology(ABC):
    """Interface for components that contain internal topology.

    The internal topology itself is intended for visualization, debugging,
    and auto-modelling scenarios; the coupling framework does NOT
    traverse it during normal simulation.
    """

    @property
    def has_internal_topology(self) -> bool:
        """Whether this component has meaningful internal topology."""
        return False

    def get_internal_nodes(self) -> list[dict]:
        """Return internal nodes for visualization.

        Each dict contains at minimum:
            {"id": str, "name": str, "type": str, ...}
        where type is one of: basin, cell, river, section, junction, ...
        Optional keys: coordinates, area, elevation, etc.
        """
        return []

    def get_internal_edges(self) -> list[dict]:
        """Return internal directed edges for visualization.

        Each dict contains:
            {"source": str, "target": str, "type": str, ...}
        where type is one of: flow, diversion, return, ...
        """
        return []

    def get_exposed_ports(self) -> list[ExchangeMeta]:
        """Return subset of internal nodes that are exposed
        as coupling ports to other components.
        """
        return []


# ---------------------------------------------------
# region ITopologyLayer
# ---------------------------------------------------


class ITopologyLayer(ABC):
    """A single layer in a potentially hierarchical topology.

    A simple lumped model has one layer (SCALAR, element_count=1).
    A distributed rainfall-runoff model may have:
        Layer 0: basin level (3 sub-basins)
        Layer 1: grid level (10 000 cells per sub-basin)
    A 1-D/2-D coupled model may have:
        Layer 0: 1-D channel network (reaches + cross-sections)
        Layer 1: 2-D floodplain grid (triangular mesh)
    """

    @property
    @abstractmethod
    def layer_id(self) -> str:
        """Unique identifier for this layer, e.g. "basins", "grid"."""
        pass

    @property
    @abstractmethod
    def parent_id(self) -> Optional[str]:
        """ID of the parent layer; None for the root layer."""
        pass

    @property
    @abstractmethod
    def spatial_dim(self) -> str:
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

    def get_parent_map(self) -> dict[str, Optional[str]]:
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
    """Fast spatial queries on the internal topology.

    Built on demand from the layer data; implementations may use
    scipy.spatial.KDTree, rtree, or custom hash grids.
    """

    @abstractmethod
    def find_nearest(self, x: float, y: float, z: float = 0.0) -> Optional[str]:
        """Return the id of the nearest element to the given coordinates."""
        pass

    @abstractmethod
    def find_containing(self, x: float, y: float, z: float = 0.0) -> Optional[str]:
        """Return the id of the element that contains the point,
        or None if the point is outside the domain."""
        pass

    @abstractmethod
    def find_within_radius(self, x: float, y: float, radius: float) -> list[str]:
        """Return ids of all elements within *radius* (m) of the point."""
        pass

    @abstractmethod
    def interpolate_to_point(
        self,
        field_values: dict[str, np.ndarray],
        x: float,
        y: float,
        z: float = 0.0,
    ) -> dict[str, float]:
        """Interpolate field values to an arbitrary point.

        Args:
            field_values: {field_name: array of length element_count}
            x, y, z: target coordinates

        Returns:
            {field_name: interpolated_value}
        """
        pass


# ---------------------------------------------------
# IInternalTopology (enhanced)
# ---------------------------------------------------


class IInternalTopology(ABC):
    """Expose a component's internal spatial structure with full
    dimension awareness and hierarchical layering.

    The coupling framework does NOT traverse this topology during
    normal time-stepping; it is used for visualisation, debugging,
    spatial queries, and auto-modelling reasoning.
    """

    @property
    def has_internal_topology(self) -> bool:
        """Whether this component has meaningful internal topology."""
        return True

    @abstractmethod
    def get_topology_meta(self) -> TopologyMeta:
        """Return a lightweight summary descriptor."""
        pass

    @abstractmethod
    def get_layers(self) -> list[ITopologyLayer]:
        """Return all topology layers, outermost first.

        A simple model returns a single layer.
        A nested basin model returns [basin_layer, grid_layer].
        A 1-D/2-D coupled model returns [channel_1d_layer, floodplain_2d_layer].
        """
        pass

    def get_layer(self, layer_id: str) -> Optional[ITopologyLayer]:
        """Convenience: fetch a layer by its id."""
        for layer in self.get_layers():
            if layer.layer_id == layer_id:
                return layer
        return None

    def get_spatial_index(self, layer_id: str = "") -> Optional[ISpatialIndex]:
        """Return a spatial index for the given layer.

        If *layer_id* is empty, the finest (innermost) layer is used.
        Returns None if the layer has no spatial extent (e.g. SCALAR).
        """
        return None

    @abstractmethod
    def get_exposed_ports(self) -> list[ExchangeMeta]:
        """Return internal nodes that are exposed as coupling ports.

        These are the only points through which data enters or leaves
        the component during coupled simulation.
        """
        pass
