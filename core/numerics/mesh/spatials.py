# -*- encoding: utf-8 -*-
"""
Copyright (C) 2026, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Spatial domain classes and methods for the cfd.
"""
from core.numerics.enums import MeshDimension, ElementType
from core.numerics.mesh.elements import Node, Face, Cell

import pickle


# -----------------------------------------------
# region Mesh
# -----------------------------------------------


class Mesh:
    """Abstract mesh class for spatial domains."""

    def __init__(self):
        self._version = 1
        self._dim = MeshDimension.NONE
        self._orthogonal = False

        self._topo = None
        self._geom = None
        self._part = None
        self._groups = {}

        self._nodes = []
        self._faces = []
        self._cells = []

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
    def nodes(self) -> list[Node]:
        """Return all nodes."""
        return self._nodes

    @property
    def face_count(self) -> int:
        """Return number of faces."""
        return len(self._faces)

    @property
    def faces(self) -> list[Face]:
        """Return all faces."""
        return self._faces

    @property
    def cell_count(self) -> int:
        """Return number of cells."""
        return len(self._cells)

    @property
    def cells(self) -> list[Cell]:
        """Return all cells."""
        return self._cells

    # -----------------------------------------------
    # methods
    # -----------------------------------------------

    def get_nodes(self, node_ids: list[int]) -> list[Node]:
        """Get the nodes with the given ids."""
        return [self._nodes[i] for i in node_ids]

    def get_faces(self, face_ids: list[int]) -> list[Face]:
        """Get the faces with the given ids."""
        return [self._faces[i] for i in face_ids]

    def get_cells(self, cell_ids: list[int]) -> list[Cell]:
        """Get the cells with the given ids."""
        return [self._cells[i] for i in cell_ids]

    def update(self, masks: list[int]):
        """Update the mesh with the given mask, for AMR.

        The mask list has the same length to mesh cells.
        Each element corresponds to a cell:

        + 1 indicates the cell should be refined.
        + 0 indicates the cell should remain unchanged.
        + -1 indicates the cell should be coarsened.
        """
        raise NotImplementedError()

    # -----------------------------------------------
    # groups
    # -----------------------------------------------

    def set_group(self, etype: ElementType, group_id: str, ids: list):
        """Set the group with the given element ids ."""
        if not isinstance(etype, ElementType):
            raise ValueError("Invalid element type.")
        if group_id in self._groups:
            raise ValueError("Group already exists.")

        min_id, max_id = min(ids), max(ids)
        if etype == ElementType.NODE:
            elem_count = self.node_count
        elif etype == ElementType.FACE:
            elem_count = self.face_count
        elif etype == ElementType.CELL:
            elem_count = self.cell_count
        else:
            raise ValueError("Element type: None.")

        if min_id < 0 or max_id >= elem_count:
            raise ValueError("Invalid group ids, out of mesh.")
        self._groups[group_id] = (ids, etype)

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
        from core.numerics.algos import MeshTopo

        if self._topo is None:
            self._topo = MeshTopo(self)
        return self._topo

    def get_geom_assistant(self):
        """Return the mesh geometry assistant."""
        from core.numerics.algos import MeshGeom

        if self._geom is None:
            self._geom = MeshGeom(self)
        return self._geom

    def get_part_assistant(self):
        """Return the mesh partition assistant."""
        from core.numerics.algos import MeshPart

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
        self._nx = None
        self._ny = None
        self._nz = None
        self._dx = None
        self._dy = None
        self._dz = None

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
    def dx(self) -> float:
        """Discretization step in the x-direction."""
        return self._dx

    @property
    def dy(self) -> float:
        """Discretization step in the y-direction."""
        return self._dy

    @property
    def dz(self) -> float:
        """Discretization step in the z-direction."""
        return self._dz

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
