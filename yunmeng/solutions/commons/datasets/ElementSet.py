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

# ---------------------------------------------------
# region SimpleElementSet
# ---------------------------------------------------


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


# ---------------------------------------------------
# region ElementSet
# ---------------------------------------------------


class ElementSet(IElementSet):
    """ElementSet class describes a collection of spatial elements.

    In general, spatial objects are fixed. Based on that, the elements are
    required to be specified when elementset is initialized,
    meanwhile only the fetch interfaces is provided.

    For changing spatial objects, such as an adaptive dynamic grid,
    the elementset is recreated or reset each time.
    """

    def __init__(self, mesh: Mesh, etype: ElementType, elements: list = None):
        # TODO: Check if all elements are of the same type.
        # NOTE: Only support `Element` objects for now.
        self._mesh = mesh
        self._etype = etype
        self._elements = self._collect_elements(elements, etype)
        self._geom_type = check_mesh_element_geom(self._elements, mesh)

    def _collect_elements(self, indices: list, element_type: ElementType):
        """Collect elements of the given type with the given indices."""
        if self._mesh is None:
            return indices

        if element_type == ElementType.NODE:
            elements = self._mesh.nodes
        elif element_type == ElementType.FACE:
            elements = self._mesh.faces
        elif element_type == ElementType.CELL:
            elements = self._mesh.cells

        if not indices:
            return elements

        objs = []
        for i in indices:
            if i < 0 or i >= len(elements):
                raise ValueError("Index out of range.")
            objs.append(elements[i])
        return objs

    @property
    def element_geom_type(self) -> GeometryType:
        return self._geom_type

    @property
    def element_count(self) -> int:
        return len(self._elements)

    def get_element_index(self, element_id: str | int) -> int:
        for i, element in enumerate(self._elements):
            eid = element_id
            if self._geom_type != GeometryType.IDBASED:
                eid = element.id
            if eid == int(element_id):
                return i
        raise None

    def get_element_id(self, element_index: int) -> str | int:
        if self._geom_type == GeometryType.IDBASED:
            return self._elements[element_index]
        return self._elements[element_index].id

    def get_face_count(self, element_index: int) -> int:
        if self._etype == ElementType.NODE:
            return 0
        elif self._etype == ElementType.FACE:
            return 1
        elif self._etype == ElementType.CELL:
            return len(self._elements[element_index].faces)
        else:
            raise None

    def get_node_count(self, element_index: int) -> int:
        if self._etype == ElementType.NODE:
            return 1
        elif self._etype == ElementType.FACE:
            return len(self._elements[element_index].nodes)
        elif self._etype == ElementType.CELL:
            topo = self._mesh.get_topo_assistant()
            cid = self._elements[element_index].id
            return len(topo.cell_nodes[cid])
        else:
            raise None

    def get_face_node_indices(self, element_index: int) -> list[int]:
        if self._etype == ElementType.NODE:
            return []
        elif self._etype == ElementType.FACE:
            return self._elements[element_index].nodes
        elif self._etype == ElementType.CELL:
            topo = self._mesh.get_topo_assistant()
            cid = self._elements[element_index].id
            return topo.cell_nodes[cid]
        else:
            raise None

    def get_node_coordinates(self, element_index: int) -> list[Coordinate]:
        if self._etype == ElementType.NODE:
            return [self._elements[element_index].coordinate]
        elif self._etype == ElementType.FACE:
            nodes = self._elements[element_index].nodes
            return [self._mesh.nodes[i].coordinate for i in nodes]
        elif self._etype == ElementType.CELL:
            topo = self._mesh.get_topo_assistant()
            cid = self._elements[element_index].id
            nodes = topo.cell_nodes[cid]
            return [self._mesh.nodes[i].coordinate for i in nodes]
        else:
            raise None

    def has_element(self, element_id: str | int) -> bool:
        for element in self._elements:
            if element.id == int(element_id):
                return True
        return False


def check_mesh_element_geom(elements: list[Element], mesh: Mesh) -> GeometryType:
    """Check the geometry type of an mesh element."""
    if mesh is None:
        return GeometryType.IDBASED

    element = elements[0]
    if isinstance(element, Node):
        return GeometryType.POINT
    elif isinstance(element, Face):
        if mesh.dimension == MeshDimension.D3:
            return GeometryType.POLYGON
        else:
            return GeometryType.POLYLINE
    elif isinstance(element, Cell):
        if mesh.dimension == MeshDimension.D3:
            return GeometryType.CUBE
        else:
            return GeometryType.POLYGON
    else:
        raise TypeError("Unsupported element type.")
