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
# region Mesh
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
