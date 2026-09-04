# -*- encoding: utf-8 -*-
"""
Copyright (C) 2026, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Mesh protocols (minimal surface) for the interfaces layer.

Only what IOperator.build / boundary providers actually need is
declared here (v2.0 §14.3). Region is deliberately NOT restricted to
the outer boundary — interior face/cell sets are legitimate regions
(v2.0 §5.6: internal constraints = Region special case).
"""

from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable

from yunmeng.interfaces.types import ArrayLike, ElementType, MeshDimension

# ---------------------------------------------------
# region Region
# ---------------------------------------------------


@runtime_checkable
class IRegion(Protocol):
    """A named set of mesh elements (faces or cells).

    May lie on the outer boundary OR in the domain interior.
    """

    @property
    def id(self) -> str: ...

    @property
    def loc(self) -> ElementType:
        """Element kind the region is defined over (usually FACE)."""
        ...

    @property
    def element_ids(self) -> ArrayLike:
        """Canonical element indices (int array, backend-moved at build)."""
        ...


# ---------------------------------------------------
# region Assistants (minimal)
# ---------------------------------------------------


@runtime_checkable
class ITopoAssistant(Protocol):
    """Topology queries needed at operator build time."""

    def neighbors(self, element: int, loc: ElementType) -> Sequence[int]: ...

    def boundary_faces(self) -> ArrayLike:
        """Canonical boundary-face numbering used by BoundaryValues
        (v2.0 §5.5). Providers and operators share this ordering."""
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
        """Topology version. AMR re-meshing bumps this; solvers trigger
        explicit operator re-build on change (v2.0 §4.2)."""
        ...

    def element_count(self, loc: ElementType) -> int: ...

    def get_topo_assistant(self) -> ITopoAssistant: ...

    def get_geom_assistant(self) -> IGeomAssistant: ...

    def get_region(self, region_id: str) -> IRegion: ...

    def regions(self) -> Sequence[IRegion]: ...
