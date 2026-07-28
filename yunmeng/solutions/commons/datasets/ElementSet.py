# -*- encoding: utf-8 -*-
"""
Copyright (C) 2025, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

ElementSet class describes a collection of spatial elements.
"""

from yunmeng.solutions.standards import IElementSet, GeometryType
from yunmeng.numerics.enums import ElementType, MeshDimension
from yunmeng.numerics.mesh import (
    Mesh,
    Element,
    Node,
    Face,
    Cell,
    Coordinate,
)
import numpy as np


class SimpleElementSet(IElementSet):
    """Mesh-free element set covering IDBASED / POINT / POLYGON footprints."""

    def __init__(
        self,
        gtype: GeometryType = GeometryType.IDBASED,
        centers: np.ndarray = None,
        outlines: list[np.ndarray] = None,
        element_ids: list[str] = None,
    ):
        """Initialize element set with given geometry type and element centers.

        Args:
            gtype: geometry type of the elements.
            centers: (n, 3) array of element centroids; for IDBASED this may
                be omitted (a single bulk element is assumed).
            outlines: optional list of (m_i, 3) vertex arrays for POLYGON
                elements.
            element_ids: optional string ids; defaults to positional indices.
        """
        self._gtype = gtype
        if gtype == GeometryType.IDBASED and centers is None:
            centers = np.zeros((1, 3))
        self._centers = np.atleast_2d(np.asarray(centers, dtype=float))
        self._outlines = outlines
        self._ids = element_ids or [str(i) for i in range(len(self._centers))]

    @property
    def element_count(self) -> int:
        return len(self._centers)

    @property
    def gtype(self) -> GeometryType:
        return self._gtype

    @property
    def element_ids(self) -> list[str]:
        return self._ids

    def index_of(self, element_id: str) -> int:
        return self._ids.index(element_id)

    def get_coordinates(self, element_index: int) -> np.ndarray:
        """Element vertices: outline if available, else the centroid."""
        if self._outlines is not None:
            return np.asarray(self._outlines[element_index], dtype=float)
        return self._centers[element_index].reshape(1, 3)

    def get_center(self, element_index: int) -> np.ndarray:
        return self._centers[element_index]

    def __repr__(self) -> str:
        return f"SimpleElementSet({self._gtype.value}, n={self.element_count})"


class ScalarElementSet(SimpleElementSet):
    """Convenience: single bulk element for lumped quantities."""

    def __init__(self, element_id: str = "0"):
        super().__init__(GeometryType.IDBASED, element_ids=[element_id])


class PointElementSet(SimpleElementSet):
    """Point set for rain / evaporation gauges."""

    def __init__(self, centers: np.ndarray, element_ids: list[str] = None):
        super().__init__(GeometryType.POINT, centers=centers, element_ids=element_ids)
