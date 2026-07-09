# -*- encoding: utf-8 -*-
"""
Copyright (C) 2026, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Spatial domain classes and methods for the cfd.
"""

from yunmeng.numerics.enums import MeshDimension, ElementType
from yunmeng.numerics.mesh.elements import Node, Face, Cell, Element

import numpy as np
from dataclasses import dataclass
from typing import Optional, Callable

# -----------------------------------------------
# region Mesh
# -----------------------------------------------


class Mesh:
    """Abstract mesh class for spatial domain."""

    def __init__(self):
        self._dim = MeshDimension.NONE
        self._orthogonal = False
        self._version = 1

        self._topo = None
        self._geom = None
        self._part = None
        self._groups = {}

        self._nodes: np.ndarray = None
        self._faces: np.ndarray = None
        self._cells: np.ndarray = None

    # -----------------------------------------------
    # properties
    # -----------------------------------------------

    @property
    def dimension(self) -> MeshDimension:
        """Return mesh dimension."""
        return self._dim

    @property
    def version(self) -> int:
        """Return the mesh version."""
        return self._version

    @property
    def orthogonal(self) -> bool:
        """Return mesh orthogonality."""
        return self._orthogonal

    @property
    def node_count(self) -> int:
        """Return number of nodes."""
        return len(self._nodes)

    @property
    def nodes(self) -> np.ndarray:
        """Return all nodes."""
        return self._nodes

    @property
    def face_count(self) -> int:
        """Return number of faces."""
        return len(self._faces)

    @property
    def faces(self) -> np.ndarray:
        """Return all faces."""
        return self._faces

    @property
    def cell_count(self) -> int:
        """Return number of cells."""
        return len(self._cells)

    @property
    def cells(self) -> np.ndarray:
        """Return all cells."""
        return self._cells

    # -----------------------------------------------
    # methods
    # -----------------------------------------------

    def get_element_count(self, etype: ElementType) -> int:
        """Return the number of elements of the given type."""
        if etype == ElementType.NODE:
            return self.node_count
        elif etype == ElementType.FACE:
            return self.face_count
        elif etype == ElementType.CELL:
            return self.cell_count
        else:
            raise ValueError("Invalid element type.")

    def get_elements(self, etype: ElementType) -> np.ndarray:
        """Return all elements of the given type."""
        if etype == ElementType.NODE:
            return self._nodes
        elif etype == ElementType.FACE:
            return self._faces
        elif etype == ElementType.CELL:
            return self._cells
        else:
            raise ValueError("Invalid element type.")

    def get_nodes(self, node_ids: list[int]) -> list[Node]:
        """Get the nodes with the given ids."""
        return self._nodes[node_ids]

    def get_faces(self, face_ids: list[int]) -> list[Face]:
        """Get the faces with the given ids."""
        return self._faces[face_ids]

    def get_cells(self, cell_ids: list[int]) -> list[Cell]:
        """Get the cells with the given ids."""
        return self._cells[cell_ids]

    # -----------------------------------------------
    # groups
    # -----------------------------------------------

    def set_group(self, group_id: str, element_ids: list, etype: ElementType):
        """Set the group with the given element ids ."""
        if not isinstance(etype, ElementType):
            raise ValueError("Invalid element type.")
        if group_id in self._groups:
            raise ValueError("Group already exists.")

        min_id, max_id = min(element_ids), max(element_ids)
        if etype == ElementType.NODE:
            elem_count = self.node_count
        elif etype == ElementType.FACE:
            elem_count = self.face_count
        elif etype == ElementType.CELL:
            elem_count = self.cell_count
        else:
            raise ValueError("Element type: None.")

        if min_id < 0 or max_id >= elem_count:
            raise ValueError("Invalid group ids.")
        self._groups[group_id] = (element_ids, etype)

    def delete_group(self, group_id: str):
        """Delete the given name group."""
        if group_id in self._groups:
            self._groups.pop(group_id)

    def has_group(self, group_id: str) -> bool:
        """Check if the group exists."""
        return group_id in self._groups

    def get_group(self, group_id: str) -> tuple[list, ElementType]:
        """Return the element ids of given group."""
        return self._groups[group_id]

    # -----------------------------------------------
    # assistants
    # -----------------------------------------------

    def get_topo_assistant(self):
        """Return the mesh topology assistant."""
        from yunmeng.numerics.algos import MeshTopo

        if self._topo is None:
            self._topo = MeshTopo(self)
        return self._topo

    def get_geom_assistant(self):
        """Return the mesh geometry assistant."""
        from yunmeng.numerics.algos import MeshGeom

        if self._geom is None:
            self._geom = MeshGeom(self)
        return self._geom

    def get_part_assistant(self):
        """Return the mesh partition assistant."""
        from yunmeng.numerics.algos import MeshPart

        if self._part is None:
            self._part = MeshPart(self)
        return self._part

    def modify(self, modifier, **kwargs):
        """Modify the mesh."""
        from yunmeng.numerics.algos import MeshModifyMode

        if modifier.validate(self, **kwargs):
            modifier.modify(self, **kwargs)
            self._version += 1
            self._geom = None
            if modifier.mode != MeshModifyMode.GEOMETRY:
                self._topo = None
                self._part = None
        else:
            raise ValueError("Invalid modifier.")


# -----------------------------------------------
# region Region
# -----------------------------------------------


@dataclass
class Region:
    """
    A region of the mesh.

    priority: indices > tags > predicate > type。
    """

    name: str
    mesh: Mesh
    type: ElementType = ElementType.NONE
    indices: Optional[list[int]] = None
    tags: Optional[list[str]] = None
    predicate: Optional[Callable[[np.ndarray], np.ndarray]] = None

    _element_ids = None
    _version = None

    def select(self, elements: np.ndarray) -> np.ndarray:
        """Return the mask of the region."""
        if self.predicate is not None:
            return self.predicate(elements)
        raise ValueError(f"Region {self.name} have no predicate method.")

    def get_element_ids(self) -> np.ndarray:
        """Get the ids of elements in the region."""
        if self._version == self.mesh.version and self._element_ids is not None:
            return self._element_ids
        if self._version != self.mesh.version:
            self._version = self.mesh.version
            self._element_ids = None

        mesh = self.mesh
        if self.indices is not None:
            resolved_ids = np.array(self.indices)
        elif self.tags is not None:
            ids, etype = mesh.get_group(self.name)
            resolved_ids = np.array(ids)
        elif self.predicate is not None:
            elements = mesh.get_elements(self.type)
            mask = self.select(elements)
            resolved_ids = np.where(mask)[0]
        elif self.type is not None:
            element_nb = mesh.get_element_count(self.type)
            resolved_ids = np.arange(element_nb)
        else:
            raise ValueError("Invalid region definition.")

        if self._element_ids is None:
            self._element_ids = resolved_ids
        return resolved_ids

    def include(self, elemnet_id: int, etype: ElementType) -> bool:
        """Check if the element is in the region."""
        if etype != self.type:
            return False

        return elemnet_id in self.get_element_ids()
