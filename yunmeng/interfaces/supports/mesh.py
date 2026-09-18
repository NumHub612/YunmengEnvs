# -*- encoding: utf-8 -*-
"""
Copyright (C) 2026, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Mesh protocols (minimal surface) for the interfaces layer.
"""

from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable
from dataclasses import dataclass

from yunmeng.interfaces.types import ArrayLike, ElementType, MeshDimension

#: Canonical neighbour direction order, truncated to the mesh dimension.
NEIGHBOUR_ORDER: tuple[str, ...] = ("-x", "+x", "-y", "+y", "-z", "+z")

#: Sentinel for a missing neighbour / outside-domain element.
TOPO_NONE: int = -1


# ---------------------------------------------------
# region Region
# ---------------------------------------------------


@dataclass(frozen=True)
class Region:
    """A named set of mesh elements (boundary or interior)."""

    name: str
    loc: ElementType
    element_ids: ArrayLike


# ---------------------------------------------------
# region Assistants
# ---------------------------------------------------


@runtime_checkable
class ITopoAssistant(Protocol):
    """Topology queries needed at operator build time."""

    def neighbors(self, element: int, loc: ElementType) -> Sequence[int]:
        """Fixed-order same-kind neighbours of `element`.

        CELL / NODE: [-x, +x, -y, +y, -z, +z] truncated to dim;NONE (-1)
        marks a missing neighbour.
        FACE: (minus_cell, plus_cell) along the face's axis; NONE marks
        the out-of-domain side of a boundary face.
        """
        ...

    def connectivity(
        self, element: int, from_loc: ElementType, to_loc: ElementType
    ) -> Sequence[int]:
        """Fixed-order incident elements of another kind.

        Supported pairs :
          CELL -> FACE: the cell's faces in [-x, +x, -y, +y,..] order.
          CELL -> NODE: the cell's vertices, low-to-high corner order
                        (bit b of corner = + along axis b).
          FACE -> NODE: the face's vertices, C-order over the face's
                        tangent axes.
          FACE -> CELL: same as neighbors(face, FACE).
          NODE -> CELL: incident cells, [-x, +x, -y, +y, ...] order
                        (the cell on each side of the node along each
                        axis); NONE outside the domain.
          NODE -> FACE: incident faces in [-x, +x, -y, +y, ...] order
                        (the face of each axis-family whose far corner
                        is the node on the minus side, near corner on
                        the plus side); NONE outside the domain.
        """
        ...

    def boundary_elements(self, loc: ElementType) -> ArrayLike:
        """Canonical indices of boundary elements."""
        ...

    def interior_elements(self, loc: ElementType) -> ArrayLike: ...


@runtime_checkable
class IGeomAssistant(Protocol):
    """Geometry queries needed at operator build time."""

    def coordinates(self, loc: ElementType) -> ArrayLike:
        """All centroids of the given kind, shape (count, 3)."""
        ...

    def centroid(self, element: int, loc: ElementType) -> ArrayLike: ...

    def face_normal(self, face: int) -> ArrayLike: ...

    def face_area(self, face: int) -> ArrayLike: ...

    def cell_volume(self, cell: int) -> ArrayLike: ...


@runtime_checkable
class IPartAssistant(Protocol):
    """Partition queries needed at operator build time."""

    def partition(self, num_shards: int, loc: ElementType) -> list: ...


# ---------------------------------------------------
# region IMesh
# ---------------------------------------------------


@runtime_checkable
class IMesh(Protocol):
    """Minimal mesh surface referenced by interface signatures."""

    @property
    def dimension(self) -> MeshDimension: ...

    @property
    def version(self) -> int: ...

    def element_count(self, loc: ElementType) -> int: ...

    def get_topo_assistant(self) -> ITopoAssistant: ...

    def get_geom_assistant(self) -> IGeomAssistant: ...

    def get_part_assistant(self) -> IPartAssistant: ...

    def get_region(self, region_id: str) -> Region: ...

    def regions(self) -> Sequence[Region]: ...


# ---------------------------------------------------
# region IGrid
# ---------------------------------------------------


@runtime_checkable
class IGrid(IMesh, Protocol):
    """Structured-grid capability (ALSO satisfies IMesh)."""

    @property
    def shape(self) -> tuple[int, ...]:
        """Logical grid shape counted in CELLs: (nx,) / (nx, ny) / (nx, ny, nz)."""
        ...

    @property
    def spacing(self) -> tuple[float, ...]:
        """Cell spacing per axis: (dx,) / (dx, dy) / (dx, dy, dz)."""
        ...

    @property
    def origin(self) -> tuple[float, ...]:
        """Coordinate of the grid's lower corner."""
        ...

    @property
    def uniform(self) -> bool:
        """Whether spacing is constant along every axis."""
        ...

    def flat_index(self, ijk: tuple[int, ...]) -> int:
        """Logical (i, j[, k]) -> flat cell index (C-order)."""
        ...

    def ijk_index(self, flat: int) -> tuple[int, ...]:
        """Flat cell index -> logical (i, j[, k])."""
        ...

    def interior_slice(self) -> tuple[slice, ...]:
        """Slice selecting interior cells on the multi-dim view,
        e.g. (slice(1, -1),) * ndim."""
        ...

    def axis_slice(self, axis: int, shift: int) -> tuple[slice, ...]:
        """Slice of the multi-dim view shifted by `shift` along
        `axis` (shift=-1 -> cells i-1, +1 -> cells i+1),
        clipped to the interior-compatible range."""
        ...
