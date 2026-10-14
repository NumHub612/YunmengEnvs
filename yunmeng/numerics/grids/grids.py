# -*- encoding: utf-8 -*-
"""
Copyright (C) 2026, The YunMengEnvs Project Contributors. Welcome aboard YunmengEnvs!

StructuredGrids.
"""

from __future__ import annotations
import numpy as np

from yunmeng.standard.types import ElementType, MeshDimension
from yunmeng.standard.supports import Region
from yunmeng.numerics.algos import AnalyticTopo, AnalyticGeom, face_count, node_count


# -----------------------------------------------
# region StructuredGrid
# -----------------------------------------------
class StructuredGrid:
    """Uniform cell-centered structured grid of dimension len(shape)."""

    def __init__(
        self,
        origin: tuple[float, ...],
        upper: tuple[float, ...],
        shape: tuple[int, ...],
    ):
        if not (len(origin) == len(upper) == len(shape)):
            raise ValueError("origin/upper/shape must have the same length.")
        if any(n < 1 for n in shape):
            raise ValueError("Cell counts must be >= 1.")
        if any(u <= o for o, u in zip(origin, upper)):
            raise ValueError("upper must be greater than origin.")

        self._origin = tuple(float(v) for v in origin)
        self._shape = tuple(int(n) for n in shape)
        self._spacing = tuple(
            (u - o) / n for o, u, n in zip(self._origin, upper, self._shape)
        )
        self._version = 1
        self._regions: dict[str, Region] = {}
        self._topo = AnalyticTopo(self)
        self._geom = AnalyticGeom(self)

    # ---------------------------------------------------------- #
    # IMesh
    # ---------------------------------------------------------- #

    @property
    def dimension(self) -> MeshDimension:
        """Mesh dimension."""
        return MeshDimension(f"{len(self._shape)}d")

    @property
    def orthogonal(self) -> bool:
        """Whether the mesh is orthogonal."""
        return True

    @property
    def version(self) -> int:
        """Mesh version."""
        return self._version

    @property
    def cell_count(self) -> int:
        """Total cell count in the grid."""
        return int(np.prod(self._shape))

    @property
    def face_count(self) -> int:
        """Total face count in the grid."""
        return face_count(self._shape)

    @property
    def node_count(self) -> int:
        """Total node count in the grid."""
        return node_count(self._shape)

    def element_count(self, loc: ElementType) -> int:
        """Total element count in the grid."""
        if loc == ElementType.CELL:
            return self.cell_count
        if loc == ElementType.FACE:
            return self.face_count
        if loc == ElementType.NODE:
            return self.node_count
        raise ValueError(f"Invalid loc: {loc}")

    def get_topo_assistant(self) -> AnalyticTopo:
        """Get the analytic topology assistant."""
        return self._topo

    def get_geom_assistant(self) -> AnalyticGeom:
        """Get the analytic geometry assistant."""
        return self._geom

    def get_region(self, region_id: str) -> Region:
        """Get the region by region_id."""
        return self._regions[region_id]

    def regions(self) -> list[Region]:
        """Get all regions in the grid."""
        return list(self._regions.values())

    def add_region(self, region: Region):
        """Add a region to the grid."""
        self._regions[region.name] = region

    # ---------------------------------------------------------- #
    # IGrid
    # ---------------------------------------------------------- #

    @property
    def shape(self) -> tuple[int, ...]:
        """Logical grid shape counted in CELLs, (nx, [ny, [nz]])."""
        return self._shape

    @property
    def spacing(self) -> tuple[float, ...]:
        """Cell spacing per axis, (dx, [dy, [dz]])."""
        return self._spacing

    @property
    def origin(self) -> tuple[float, ...]:
        """Coordinates of the minimum-corner, (x0, [y0, [z0]])."""
        return self._origin

    @property
    def uniform(self) -> bool:
        """Whether the grid is uniform."""
        return True

    def flat_index(self, ijk: tuple[int, ...]) -> int:
        """Logical (i, j[, k]) -> flat cell index (C-order)
        e.g. flat = i * (ny * nz) + j * nz + k."""
        return int(np.ravel_multi_index(ijk, self._shape))

    def ijk_index(self, flat: int) -> tuple[int, ...]:
        """Flat cell index -> logical (i, j[, k])
        i = flat // (ny * nz),
        j = (flat // nz) % ny,
        k = flat % nz."""
        return tuple(int(v) for v in np.unravel_index(flat, self._shape))

    def interior_slice(self) -> tuple[slice, ...]:
        """Slice object for the interior cells of the grid
        e.g. (slice(1, -1),) * ndim."""
        return (slice(1, -1),) * len(self._shape)

    def axis_slice(self, axis: int, shift: int) -> tuple[slice, ...]:
        """Slice object for the cells shifted along a given axis,
        clipped to the interior-compatible range.

        Args:
            axis: The axis along which to shift ([0, 1, 2]).
            shift: The shift (-1 for left/down/back, +1 for opposite).
        Returns:
            A tuple of slice objects for each axis.
        """
        if not 0 <= axis < len(self._shape):
            raise ValueError(f"Invalid axis: {axis}")
        if shift == -1:
            s = slice(0, -2)  # [0:-2]
        elif shift == +1:
            s = slice(2, None)  # [2:]
        else:
            raise ValueError("shift must be -1 or +1.")
        slices = list(self.interior_slice())
        slices[axis] = s
        return tuple(slices)

    # ---------------------------------------------------------- #
    # vectorized helpers
    # ---------------------------------------------------------- #

    def cell_centers(self) -> tuple[np.ndarray, ...]:
        """Per-axis cell-center arrays, each of shape == self.shape."""
        axes = [
            o + (np.arange(n) + 0.5) * d
            for o, d, n in zip(self._origin, self._spacing, self._shape)
        ]
        return tuple(np.meshgrid(*axes, indexing="ij"))

    def boundary_faces(self) -> np.ndarray:
        return self._topo.boundary_elements(ElementType.FACE)

    def interior_faces(self) -> np.ndarray:
        return self._topo.interior_elements(ElementType.FACE)

    def node_coordinates(self) -> np.ndarray:
        """(node_count, 3) vertex coordinates."""
        return self._geom.coordinates(ElementType.NODE)


# -----------------------------------------------
# region Grid1D
# -----------------------------------------------
class Grid1D(StructuredGrid):
    """Uniform cell-centered 1D grid on [x0, x1] with nx cells."""

    def __init__(self, x0: float, x1: float, nx: int):
        """Initialize a 1D grid.

        Args:
            x0: The x-coordinate of the lower-left corner.
            x1: The x-coordinate of the upper-right corner.
            nx: The number of cells in the grid.
        """
        super().__init__((x0,), (x1,), (nx,))


# -----------------------------------------------
# region Grid2D
# -----------------------------------------------
class Grid2D(StructuredGrid):
    """Uniform cell-centered 2D grid on [x0,x1]x[y0,y1], nx*ny cells."""

    def __init__(
        self,
        lower_left: tuple[float, float] = (0.0, 0.0),
        upper_right: tuple[float, float] = (1.0, 1.0),
        nx: int = 32,
        ny: int = 32,
    ):
        """Initialize a 2D grid.

        Args:
            lower_left: The (x0, y0) coordinates of the lower-left corner.
            upper_right: The (x1, y1) coordinates of the upper-right corner.
            nx: The number of cells along the x-axis.
            ny: The number of cells along the y-axis.
        """
        super().__init__(lower_left, upper_right, (nx, ny))


# -----------------------------------------------
# region Grid3D
# -----------------------------------------------
class Grid3D(StructuredGrid):
    """Uniform cell-centered 3D grid, nx*ny*nz cells."""

    def __init__(
        self,
        lower_left: tuple[float, float, float] = (0.0, 0.0, 0.0),
        upper_right: tuple[float, float, float] = (1.0, 1.0, 1.0),
        nx: int = 16,
        ny: int = 16,
        nz: int = 16,
    ):
        """Initialize a 3D grid.

        Args:
            lower_left: The (x0, y0, z0) coordinates of the lower-left corner.
            upper_right: The (x1, y1, z1) coordinates of the upper-right corner.
            nx: The number of cells along the x-axis.
            ny: The number of cells along the y-axis.
            nz: The number of cells along the z-axis.
        """
        super().__init__(lower_left, upper_right, (nx, ny, nz))
