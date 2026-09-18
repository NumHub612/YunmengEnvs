# -*- encoding: utf-8 -*-
"""
Copyright (C) 2026, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Unified analytic topo/geom/part assistants for structured grids of ANY dim.
"""

from __future__ import annotations

import numpy as np

from yunmeng.interfaces.types import ElementType
from yunmeng.interfaces.supports import (
    TOPO_NONE,
    IGrid,
    IGeomAssistant,
    ITopoAssistant,
    IPartAssistant,
)

# ---------------------------------------------------
# region Helper functions
# ---------------------------------------------------


def _family_shape(cell_shape: tuple[int, ...], axis: int) -> tuple[int, ...]:
    s = list(cell_shape)
    s[axis] += 1
    return tuple(s)


def _family_offset(cell_shape: tuple[int, ...], axis: int) -> int:
    return sum(int(np.prod(_family_shape(cell_shape, a))) for a in range(axis))


def face_family_size(cell_shape: tuple[int, ...], axis: int) -> int:
    return int(np.prod(_family_shape(cell_shape, axis)))


def face_count(cell_shape: tuple[int, ...]) -> int:
    return _family_offset(cell_shape, len(cell_shape))


def node_count(cell_shape: tuple[int, ...]) -> int:
    return int(np.prod(tuple(n + 1 for n in cell_shape)))


def face_to_axis_plane(
    cell_shape: tuple[int, ...], face: int
) -> tuple[int, tuple[int, ...]]:
    """Flat face id -> (axis, plane multi-index within the family)."""
    for a in range(len(cell_shape)):
        off = _family_offset(cell_shape, a)
        size = face_family_size(cell_shape, a)
        if face < off + size:
            plane = tuple(
                int(v)
                for v in np.unravel_index(face - off, _family_shape(cell_shape, a))
            )
            return a, plane
    raise IndexError(f"Invalid face id: {face}")


def node_shape(cell_shape: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(n + 1 for n in cell_shape)


# ---------------------------------------------------
# region AnalyticTopo
# ---------------------------------------------------


class AnalyticTopo:
    """ITopoAssistant for any structured grid (dim-agnostic)."""

    def __init__(self, grid: IGrid):
        self._g = grid
        self._cache: dict = {}

    # -- neighbours ---------------------------------

    def _grid_neighbors(self, element: int, shape: tuple[int, ...]) -> list[int]:
        """Fixed-order [-x,+x,-y,+y,-z,+z] neighbours on a C-order lattice."""
        idx = np.unravel_index(element, shape)
        out = []
        for a, n in enumerate(shape):
            for d in (-1, 1):
                j = idx[a] + d
                if 0 <= j < n:
                    nb = list(idx)
                    nb[a] = j
                    out.append(int(np.ravel_multi_index(nb, shape)))
                else:
                    out.append(TOPO_NONE)
        return out

    def neighbors(self, element: int, loc: ElementType) -> list[int]:
        shape = self._g.shape
        if loc == ElementType.CELL:
            return self._grid_neighbors(element, shape)
        if loc == ElementType.NODE:
            return self._grid_neighbors(element, node_shape(shape))
        if loc == ElementType.FACE:
            a, plane = face_to_axis_plane(shape, element)
            out = []
            for j in (plane[a] - 1, plane[a]):
                if 0 <= j < shape[a]:
                    c = list(plane)
                    c[a] = j
                    out.append(int(np.ravel_multi_index(c, shape)))
                else:
                    out.append(TOPO_NONE)
            return out  # (minus_cell, plus_cell), length 2
        raise ValueError(f"Unsupported loc: {loc}")

    # -- cross-kind incidence -----------------------

    def _face_id(self, axis: int, plane: tuple[int, ...]) -> int:
        """Flat id of the face (axis, plane multi-index)."""
        return _family_offset(self._g.shape, axis) + int(
            np.ravel_multi_index(plane, _family_shape(self._g.shape, axis))
        )

    def _cell_faces(self, cell: int) -> list[int]:
        """Cell's faces in [-x, +x, -y, +y, ...] order."""
        shape = self._g.shape
        c = np.unravel_index(cell, shape)
        out = []
        for a in range(len(shape)):
            minus = list(c)  # plane[a] = c[a]
            out.append(self._face_id(a, tuple(minus)))
            plus = list(c)
            plus[a] += 1  # plane[a] = c[a] + 1
            out.append(self._face_id(a, tuple(plus)))
        return out

    def _cell_nodes(self, cell: int) -> list[int]:
        """Cell's vertices, corner bits low-to-high per axis."""
        shape = self._g.shape
        c = np.unravel_index(cell, shape)
        nshape = node_shape(shape)
        out = []
        for corner in range(2 ** len(shape)):
            nidx = tuple(c[a] + ((corner >> a) & 1) for a in range(len(shape)))
            out.append(int(np.ravel_multi_index(nidx, nshape)))
        return out

    def _face_nodes(self, face: int) -> list[int]:
        """Face's vertices, C-order over the face's tangent axes."""
        a, plane = face_to_axis_plane(self._g.shape, face)
        nshape = node_shape(self._g.shape)
        tangents = [b for b in range(len(plane)) if b != a]
        out = []
        for bits in range(2 ** len(tangents)):
            nidx = list(plane)
            for k, b in enumerate(tangents):
                nidx[b] += (bits >> k) & 1
            out.append(int(np.ravel_multi_index(tuple(nidx), nshape)))
        return out

    def _node_cells(self, node: int) -> list[int]:
        """Cells incident to a node, [-x, +x, -y, +y, ...] order:
        the cell on each side of the node along each axis."""
        shape = self._g.shape
        idx = np.unravel_index(node, node_shape(shape))
        out = []
        for a, n in enumerate(shape):
            for d in (-1, 0):  # minus side: idx[a]-1, plus side: idx[a]
                j = idx[a] + d
                if 0 <= j < n:
                    c = [min(idx[b], shape[b] - 1) for b in range(len(shape))]
                    c[a] = j
                    out.append(int(np.ravel_multi_index(c, shape)))
                else:
                    out.append(TOPO_NONE)
        return out

    def _node_faces(self, node: int) -> list[int]:
        """Faces incident to a node, [-x, +x, -y, +y, ...] order:
        for axis a, the family-a face whose plane index equals the node
        index along a, with tangent coordinates idx-1 (minus) / idx
        (plus)."""
        shape = self._g.shape
        idx = np.unravel_index(node, node_shape(shape))
        out = []
        for a in range(len(shape)):
            for d in (-1, 0):
                plane = list(idx)
                ok = True
                for b in range(len(shape)):
                    if b == a:
                        continue  # plane[a] = idx[a] always valid
                    plane[b] = idx[b] + d
                    if not 0 <= plane[b] < shape[b]:
                        ok = False
                out.append(self._face_id(a, tuple(plane)) if ok else TOPO_NONE)
        return out

    def connectivity(
        self, element: int, from_loc: ElementType, to_loc: ElementType
    ) -> list[int]:
        if from_loc == ElementType.CELL and to_loc == ElementType.FACE:
            return self._cell_faces(element)
        if from_loc == ElementType.CELL and to_loc == ElementType.NODE:
            return self._cell_nodes(element)
        if from_loc == ElementType.FACE and to_loc == ElementType.NODE:
            return self._face_nodes(element)
        if from_loc == ElementType.FACE and to_loc == ElementType.CELL:
            return self.neighbors(element, ElementType.FACE)
        if from_loc == ElementType.NODE and to_loc == ElementType.CELL:
            return self._node_cells(element)
        if from_loc == ElementType.NODE and to_loc == ElementType.FACE:
            return self._node_faces(element)
        if from_loc == to_loc:
            return self.neighbors(element, from_loc)
        raise ValueError(f"Unsupported pair: {from_loc} -> {to_loc}")

    # -- boundary / interior sets -------------------

    def boundary_elements(self, loc: ElementType) -> np.ndarray:
        key = ("b", loc)
        if key not in self._cache:
            self._classify(loc)
        return self._cache[key]

    def interior_elements(self, loc: ElementType) -> np.ndarray:
        key = ("i", loc)
        if key not in self._cache:
            self._classify(loc)
        return self._cache[key]

    def _classify(self, loc: ElementType):
        g = self._g
        if loc == ElementType.CELL:
            shape = g.shape
            bmask = np.zeros(shape, dtype=bool)
            for a, n in enumerate(shape):
                lo = [slice(None)] * len(shape)
                lo[a] = 0
                bmask[tuple(lo)] = True
                hi = [slice(None)] * len(shape)
                hi[a] = n - 1
                bmask[tuple(hi)] = True
        elif loc == ElementType.NODE:
            shape = node_shape(g.shape)
            bmask = np.zeros(shape, dtype=bool)
            for a, n in enumerate(shape):
                lo = [slice(None)] * len(shape)
                lo[a] = 0
                bmask[tuple(lo)] = True
                hi = [slice(None)] * len(shape)
                hi[a] = n - 1
                bmask[tuple(hi)] = True
        elif loc == ElementType.FACE:
            bnd, itf = [], []
            for a, n in enumerate(g.shape):
                off = _family_offset(g.shape, a)
                fshape = _family_shape(g.shape, a)
                planes = np.array(
                    np.unravel_index(np.arange(int(np.prod(fshape))), fshape)
                )
                is_b = (planes[a] == 0) | (planes[a] == n)
                bnd.append(off + np.flatnonzero(is_b))
                itf.append(off + np.flatnonzero(~is_b))
            self._cache[("b", loc)] = np.concatenate(bnd)
            self._cache[("i", loc)] = np.concatenate(itf)
            return
        else:
            raise ValueError(f"Unsupported loc: {loc}")

        flat = np.arange(int(np.prod(shape)))
        self._cache[("b", loc)] = flat[bmask.ravel()]
        self._cache[("i", loc)] = flat[(~bmask).ravel()]


# ---------------------------------------------------
# region AnalyticGeom
# ---------------------------------------------------


class AnalyticGeom:
    """IGeomAssistant for any structured grid (dim-agnostic)."""

    def __init__(self, grid: IGrid):
        self._g = grid
        self._coord_cache: dict = {}

    # -- helpers ------------------------------------

    def _point(self, idx: tuple[int, ...], face_axis: int) -> np.ndarray:
        origin, spacing = self._g.origin, self._g.spacing
        p = np.zeros(3)
        for a in range(len(origin)):
            if a == face_axis:
                p[a] = origin[a] + idx[a] * spacing[a]
            else:
                p[a] = origin[a] + (idx[a] + 0.5) * spacing[a]
        return p

    def _all_points(self, shape: tuple[int, ...], face_axis: int) -> np.ndarray:
        """(count, 3) coordinates of a C-order lattice of `shape`."""
        origin, spacing = self._g.origin, self._g.spacing
        axes = []
        for a, n in enumerate(shape):
            if a == face_axis:
                axes.append(origin[a] + spacing[a] * np.arange(n))
            else:
                axes.append(origin[a] + spacing[a] * (np.arange(n) + 0.5))
        grids = np.meshgrid(*axes, indexing="ij")
        pts = np.zeros((int(np.prod(shape)), 3))
        for a, ga in enumerate(grids):
            pts[:, a] = ga.ravel()
        return pts

    # -- protocol -----------------------------------

    def centroid(self, element: int, loc: ElementType) -> np.ndarray:
        shape = self._g.shape
        if loc == ElementType.CELL:
            idx = tuple(int(v) for v in np.unravel_index(element, shape))
            return self._point(idx, None)
        if loc == ElementType.NODE:
            idx = tuple(int(v) for v in np.unravel_index(element, node_shape(shape)))
            origin, spacing = self._g.origin, self._g.spacing
            p = np.zeros(3)
            for a in range(len(origin)):
                p[a] = origin[a] + idx[a] * spacing[a]
            return p
        if loc == ElementType.FACE:
            a, plane = face_to_axis_plane(shape, element)
            return self._point(plane, a)
        raise ValueError(f"Unsupported loc: {loc}")

    def coordinates(self, loc: ElementType) -> np.ndarray:
        """All centroids of the given kind, shape (count, 3)."""
        if loc not in self._coord_cache:
            shape = self._g.shape
            if loc == ElementType.CELL:
                pts = self._all_points(shape, None)
            elif loc == ElementType.NODE:
                origin, spacing = self._g.origin, self._g.spacing
                axes = [
                    origin[a] + spacing[a] * np.arange(n)
                    for a, n in enumerate(node_shape(shape))
                ]
                grids = np.meshgrid(*axes, indexing="ij")
                pts = np.zeros((node_count(shape), 3))
                for a, ga in enumerate(grids):
                    pts[:, a] = ga.ravel()
            elif loc == ElementType.FACE:
                parts = [
                    self._all_points(_family_shape(shape, a), a)
                    for a in range(len(shape))
                ]
                pts = np.concatenate(parts, axis=0)
            else:
                raise ValueError(f"Unsupported loc: {loc}")
            self._coord_cache[loc] = pts
        return self._coord_cache[loc]

    def face_normal(self, face: int) -> np.ndarray:
        a, _ = face_to_axis_plane(self._g.shape, face)
        n = np.zeros(3)
        n[a] = 1.0
        return n

    def face_area(self, face: int) -> np.ndarray:
        a, _ = face_to_axis_plane(self._g.shape, face)
        area = 1.0
        for b, s in enumerate(self._g.spacing):
            if b != a:
                area *= s
        return np.array(area)

    def cell_volume(self, cell: int) -> np.ndarray:
        return np.array(float(np.prod(self._g.spacing)))

    def cell2cell_distance(self, cell: int) -> np.ndarray:
        """Distances to neighbours in neighbors() order; NaN for TOPO_NONE."""
        nbs = self._g.get_topo_assistant().neighbors(cell, ElementType.CELL)
        spacing = self._g.spacing
        out = np.full(len(nbs), np.nan)
        for k, nb in enumerate(nbs):
            if nb != TOPO_NONE:
                out[k] = spacing[k // 2]
        return out

    def cell2face_distance(self, cell: int) -> np.ndarray:
        """Centroid-to-face distances, in connectivity(CELL->FACE) order."""
        ndim = len(self._g.shape)
        out = np.empty(2 * ndim)
        for a in range(ndim):
            out[2 * a] = self._g.spacing[a] / 2
            out[2 * a + 1] = self._g.spacing[a] / 2
        return out


# ---------------------------------------------------
# region AnalyticPart
# ---------------------------------------------------


class AnalyticPart:

    def partition(self, num_shards: int) -> list:
        return [slice(None)] * 3
