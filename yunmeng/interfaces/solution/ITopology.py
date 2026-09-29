# -*- encoding: utf-8 -*-
"""
Copyright (C) 2026, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Spatial-query protocols (Dimension-aware internal topology).

Design decision recorded here:

1. The framework does NOT abstract a model's internal physical-component
   structure (basin trees, channel networks, land-surface patches).
   That structure IS the physics — it determines the algorithms — so a
   generic layer degenerates into either an untyped data bag or a
   re-implementation of each domain.

2. What remains at this layer is the spatial-query contract already
   promised to the exchange layer: POINT-type ports interpolate via
   ISpatialIndex.

3. Two-phase lifecycle, IOperator-style: build() runs ONCE per element
   set (index structures, device placement); the query methods run at
   runtime and must stay allocation-lean.
"""

from __future__ import annotations
from abc import ABC, abstractmethod

from yunmeng.interfaces.types import ArrayLike

# ---------------------------------------------------
# region ISpatialIndex
# ---------------------------------------------------


class ISpatialIndex(ABC):
    """Fast spatial queries over a fixed set of located elements.

    The index is bound to its element set at build() time; queries
    return ELEMENT IDS from that set, never coordinates — consumers map
    ids back to values themselves.

    Element ids are int64, matching Region.element_ids and the
    framework-wide integer mesh addressing (scatter/gather);
    string names belong to named entities.
    """

    # -- build phase (once per element set) ---------

    @abstractmethod
    def build(self, element_ids: ArrayLike, coords: ArrayLike):
        """Index (n, 3) coordinates under the given int64 ids. Called
        once; rebuilding on the same instance replaces the previous
        index."""
        ...

    # -- query phase (runtime) ----------------------

    @abstractmethod
    def find_containing(self, x: float, y: float, z: float = 0.0) -> int:
        """Id of the element containing the point. Raises KeyError
        when the point lies outside every element."""
        ...

    @abstractmethod
    def find_nearest(self, x: float, y: float, z: float = 0.0) -> int:
        """Id of the element nearest to the point."""
        ...

    @abstractmethod
    def find_within_radius(
        self,
        x: float,
        y: float,
        radius: float,
    ) -> ArrayLike:
        """Ids of all elements whose location lies within radius."""
        ...

    @abstractmethod
    def interpolate_to_point(
        self,
        values: dict[str, ArrayLike],
        x: float,
        y: float,
        z: float = 0.0,
        method: str = "idw",
    ) -> dict[str, float]:
        """Interpolate named value arrays(keyed by quantity, aligned
        with the indexed element order) to the query point.

        method is explicit: "nearest" |"idw" |"linear". Indices that
        cannot honor a method on their geometry (e.g. "linear" on a
        point cloud) raise ValueError rather than silently degrading.
        """
        ...
