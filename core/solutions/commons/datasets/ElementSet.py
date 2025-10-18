# -*- encoding: utf-8 -*-
"""
Copyright (C) 2025, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

ElementSet class describes a collection of spatial elements.
"""
from core.solutions.standards import IElementSet
from core.numerics.types import GeomType
from core.numerics.mesh import (
    Network,
    Mesh,
    ElementType,
    Element,
    Node,
    Face,
    Cell,
    Coordinate,
)

from typing import Any


class ElementSet(IElementSet):
    """ElementSet class describes a collection of spatial elements.

    In general, spatial objects are fixed. Based on that, the elements are
    required to be specified when elementset is initialized,
    meanwhile only the fetch interfaces is provided.

    For changing spatial objects, such as an adaptive dynamic grid,
    the elementset is recreated or reset each time.
    """

    def __init__(self, mesh: Mesh, etype: ElementType, elements: list[int]):
        # TODO: Check if all elements are of the same type.
        # NOTE: Only support `Element` objects for now.
        self._mesh = mesh
        self._etype = etype
        self._elements = self._collect_elements(elements, etype)
        self._geom_type = check_mesh_element_geom(self._elements[0], mesh)

    def _collect_elements(self, indices: list[int], element_type: ElementType):
        """Collect elements of the given type with the given indices."""
        if element_type == ElementType.NODE:
            elements = self._mesh.nodes
        elif element_type == ElementType.FACE:
            elements = self._mesh.faces
        elif element_type == ElementType.CELL:
            elements = self._mesh.cells

        objs = []
        for i in indices:
            if i < 0 or i >= len(elements):
                raise ValueError("Index out of range.")
            objs.append(elements[i])
        return objs

    @property
    def element_geom_type(self) -> GeomType:
        return self._geom_type

    @property
    def element_count(self) -> int:
        return len(self._elements)

    def get_element_index(self, element_id: str | int) -> int:
        for i, element in enumerate(self._elements):
            if element.id == int(element_id):
                return i
        raise ValueError("Element not found.")

    def get_element_id(self, element_index: int) -> str | int:
        return self._elements[element_index].id

    def get_face_count(self, element_index: int) -> int:
        if self._etype == ElementType.NODE:
            return 0
        elif self._etype == ElementType.FACE:
            return 1
        elif self._etype == ElementType.CELL:
            return len(self._elements[element_index].faces)
        else:
            raise ValueError("Unsupported element type.")

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
            raise ValueError("Unsupported element type.")

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
            raise ValueError("Unsupported element type.")

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
            raise ValueError("Unsupported element type.")


def check_mesh_element_geom(element: Element, mesh: Mesh) -> GeomType:
    """Check the geometry type of an mesh element."""
    if isinstance(element, Node):
        return GeomType.Point
    elif isinstance(element, Face):
        if mesh.dimension.value == "3d":
            return GeomType.Polygon
        else:
            return GeomType.Polyline
    elif isinstance(element, Cell):
        if mesh.dimension.value == "3d":
            return GeomType.Polyhedron
        else:
            return GeomType.Polygon
    else:
        raise TypeError("Unsupported element type.")
