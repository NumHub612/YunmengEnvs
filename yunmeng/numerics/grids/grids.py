# -*- encoding: utf-8 -*-
"""StructuredGrid1D: cell-centered structured 1D grid (IMesh + IGrid)."""

from __future__ import annotations
import numpy as np

from yunmeng.interfaces.types import ElementType, MeshDimension
from yunmeng.interfaces.supports import Region
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
    ) -> None:
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
        return MeshDimension(len(self._shape))

    @property
    def version(self) -> int:
        return self._version

    @property
    def cell_count(self) -> int:
        return int(np.prod(self._shape))

    @property
    def face_count(self) -> int:
        return face_count(self._shape)

    @property
    def node_count(self) -> int:
        """Grid vertices: (n+1) per axis."""
        return node_count(self._shape)

    def element_count(self, loc: ElementType) -> int:
        if loc == ElementType.CELL:
            return self.cell_count
        if loc == ElementType.FACE:
            return self.face_count
        if loc == ElementType.NODE:
            return self.node_count
        raise ValueError(f"Invalid loc: {loc}")

    def get_topo_assistant(self) -> AnalyticTopo:
        return self._topo

    def get_geom_assistant(self) -> AnalyticGeom:
        return self._geom

    def get_region(self, region_id: str) -> Region:
        return self._regions[region_id]

    def regions(self) -> list[Region]:
        return list(self._regions.values())

    def add_region(self, region: Region) -> None:
        self._regions[region.name] = region

    # ---------------------------------------------------------- #
    # IGrid
    # ---------------------------------------------------------- #

    @property
    def shape(self) -> tuple[int, ...]:
        """Logical grid shape counted in CELLs."""
        return self._shape

    @property
    def spacing(self) -> tuple[float, ...]:
        return self._spacing

    @property
    def origin(self) -> tuple[float, ...]:
        return self._origin

    @property
    def uniform(self) -> bool:
        return True

    def flat_index(self, ijk: tuple[int, ...]) -> int:
        return int(np.ravel_multi_index(ijk, self._shape))

    def ijk_index(self, flat: int) -> tuple[int, ...]:
        return tuple(int(v) for v in np.unravel_index(flat, self._shape))

    def interior_slice(self) -> tuple[slice, ...]:
        return (slice(1, -1),) * len(self._shape)

    def axis_slice(self, axis: int, shift: int) -> tuple[slice, ...]:
        if not 0 <= axis < len(self._shape):
            raise ValueError(f"Invalid axis: {axis}")
        if shift == -1:
            s = slice(0, -2)
        elif shift == +1:
            s = slice(2, None)
        else:
            raise ValueError("shift must be -1 or +1.")
        slices = list(self.interior_slice())
        slices[axis] = s
        return tuple(slices)

    # ---------------------------------------------------------- #
    # vectorized helpers (beyond the minimal protocol)
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
# region StructuredGrid1D
# -----------------------------------------------
class StructuredGrid1D(StructuredGrid):
    """Uniform cell-centered 1D grid on [x0, x1] with nx cells."""

    def __init__(self, x0: float, x1: float, nx: int) -> None:
        super().__init__((x0,), (x1,), (nx,))


# -----------------------------------------------
# region StructuredGrid2D
# -----------------------------------------------
class StructuredGrid2D(StructuredGrid):
    """Uniform cell-centered 2D grid on [x0,x1]x[y0,y1], nx*ny cells."""

    def __init__(
        self,
        lower_left: tuple[float, float] = (0.0, 0.0),
        upper_right: tuple[float, float] = (1.0, 1.0),
        nx: int = 32,
        ny: int = 32,
    ) -> None:
        super().__init__(lower_left, upper_right, (nx, ny))


# -----------------------------------------------
# region StructuredGrid3D
# -----------------------------------------------
class StructuredGrid3D(StructuredGrid):
    """Uniform cell-centered 3D grid, nx*ny*nz cells."""

    def __init__(
        self,
        lower_left: tuple[float, float, float] = (0.0, 0.0, 0.0),
        upper_right: tuple[float, float, float] = (1.0, 1.0, 1.0),
        nx: int = 16,
        ny: int = 16,
        nz: int = 16,
    ) -> None:
        super().__init__(lower_left, upper_right, (nx, ny, nz))
