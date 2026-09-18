# -*- encoding: utf-8 -*-
"""
Copyright (C) 2026, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Mesh protocols (minimal surface) for the interfaces layer.
"""

from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable

from yunmeng.interfaces.types import ArrayLike, ElementType, MeshDimension

# ---------------------------------------------------
# region Region
# ---------------------------------------------------


@runtime_checkable
class IRegion(Protocol):
    """A named set of mesh elements.

    May lie on the outer boundary OR in the domain interior.
    """

    @property
    def id(self) -> str: ...

    @property
    def loc(self) -> ElementType:
        """Element kind the region is defined over."""
        ...

    @property
    def element_ids(self) -> ArrayLike:
        """Canonical element indices."""
        ...


# ---------------------------------------------------
# region Assistants
# ---------------------------------------------------


@runtime_checkable
class ITopoAssistant(Protocol):
    """Topology queries needed at operator build time."""

    def neighbors(self, element: int, loc: ElementType) -> Sequence[int]: ...

    def boundary_faces(self) -> ArrayLike:
        """Canonical boundary-face numbering used by BoundaryValues."""
        ...

    def interior_faces(self) -> ArrayLike: ...


@runtime_checkable
class IGeomAssistant(Protocol):
    """Geometry queries needed at operator build time."""

    def centroid(self, element: int, loc: ElementType) -> ArrayLike: ...

    def face_normal(self, face: int) -> ArrayLike: ...

    def face_area(self, face: int) -> ArrayLike: ...

    def cell_volume(self, cell: int) -> ArrayLike: ...


# ---------------------------------------------------
# region IMesh
# ---------------------------------------------------


@runtime_checkable
class IMesh(Protocol):
    """Minimal mesh surface referenced by interface signatures."""

    @property
    def dimension(self) -> MeshDimension: ...

    @property
    def version(self) -> int:
        """Topology version. AMR re-meshing bumps this."""
        ...

    def element_count(self, loc: ElementType) -> int: ...

    def get_topo_assistant(self) -> ITopoAssistant: ...

    def get_geom_assistant(self) -> IGeomAssistant: ...

    def get_region(self, region_id: str) -> IRegion: ...

    def regions(self) -> Sequence[IRegion]: ...


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
        """Slice of the multi-dim view shifted by `shift` along `axis`
        (shift=-1 -> cells i-1, +1 -> cells i+1), clipped to the
        interior-compatible range."""
        ...
