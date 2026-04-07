# -*- encoding: utf-8 -*-
"""
Copyright (C) 2026, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Spatial domain classes and methods for the cfd.
"""
from yunmeng.numerics.enums import MeshDimension, ElementType
from yunmeng.numerics.mesh.elements import Node, Face, Cell

import numpy as np
from enum import Enum, auto
from abc import abstractmethod


# -----------------------------------------------
# region Modifier
# -----------------------------------------------


class MeshModifyMode(Enum):
    """Mesh update modes."""

    GEOMETRY = auto()  # Geometry changes (Moving mesh, deformation)
    TOPOLOGY = auto()  # Topology changes (AMR, remeshing)
    HYBRID = auto()  # Both topology and geometry change


class MeshModifier:
    """Abstract class for mesh modification operations."""

    @property
    @abstractmethod
    def mode(self) -> MeshModifyMode:
        """The modification mode."""
        pass

    @abstractmethod
    def validate(self, mesh: "Mesh", **kwargs) -> bool:
        """Validate if the modification can be applied."""
        pass

    @abstractmethod
    def modify(self, mesh: "Mesh", **kwargs):
        """Apply the modification to the mesh."""
        pass


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

    def modify(self, modifier: MeshModifier, **kwargs):
        """Modify the mesh."""
        if modifier.validate(self, **kwargs):
            modifier.modify(self, **kwargs)
            self._version += 1
            self._geom = None
            if modifier.mode != MeshModifyMode.GEOMETRY:
                self._topo = None
                self._part = None
        else:
            raise ValueError("Invalid modifier.")

    def get_topo_assistant(self):
        """Return the mesh topology assistant."""
        from yunmeng.numerics.algos.topos import MeshTopo

        if self._topo is None:
            self._topo = MeshTopo(self)
        return self._topo

    def get_geom_assistant(self):
        """Return the mesh geometry assistant."""
        from yunmeng.numerics.algos.geoms import MeshGeom

        if self._geom is None:
            self._geom = MeshGeom(self)
        return self._geom

    def get_part_assistant(self):
        """Return the mesh partition assistant."""
        from yunmeng.numerics.algos.parts import MeshPart

        if self._part is None:
            self._part = MeshPart(self)
        return self._part


# -----------------------------------------------
# region Grid
# -----------------------------------------------


class Grid(Mesh):
    """Abstract class for orthogonal structured grids."""

    def __init__(self):
        super().__init__()
        self._orthogonal = True
        self._uniform = False
        self._nx = None
        self._ny = None
        self._nz = None
        self._lx = None
        self._ly = None
        self._lz = None

    # -----------------------------------------------
    # properties
    # -----------------------------------------------

    @property
    def nx(self) -> int:
        """Discretization size in the x-direction."""
        return self._nx

    @property
    def ny(self) -> int:
        """Discretization size in the y-direction."""
        return self._ny

    @property
    def nz(self) -> int:
        """Discretization size in the z-direction."""
        return self._nz

    @property
    def uniform(self) -> bool:
        """Return if the grid is uniform."""
        return self._uniform

    @property
    def lx(self) -> float:
        """Length of the grid in the x-direction."""
        return self._lx

    @property
    def ly(self) -> float:
        """Length of the grid in the y-direction."""
        return self._ly

    @property
    def lz(self) -> float:
        """Length of the grid in the z-direction."""
        return self._lz

    # -----------------------------------------------
    # methods
    # -----------------------------------------------

    def match_node(self, i: int, j: int, k: int) -> int:
        """Match node with the local indices."""
        raise NotImplementedError()

    def match_cell(self, i: int, j: int, k: int) -> int:
        """Match cell with the local indices."""
        raise NotImplementedError()

    def get_node_neighbours(self, id: int) -> list[int]:
        """Get the neighbours node indices, sorted in:
        [east, west, north, south, top, bottom]
        """
        raise NotImplementedError()

    def get_cell_neighbours(self, id: int) -> list[int]:
        """Get the neighbours cell indices, sorted in:
        [east, west, north, south, top, bottom]
        """
        raise NotImplementedError()


# -----------------------------------------------
# region Network
# -----------------------------------------------


class Network:
    """
    Abstract network class for topological connectivity.
    """

    def to_mesh(self) -> Mesh:
        """
        Convert network to mesh.
        """
        raise NotImplementedError()
