# -*- encoding: utf-8 -*-
"""
Copyright (C) 2025, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Mesh topology processing.
"""

from yunmeng.numerics.enums import MeshDimension
from yunmeng.numerics.mesh import Mesh, sort_anticlockwise

from typing import List
import collections
import numpy as np

# -----------------------------------------------
# region MeshTopo
# -----------------------------------------------


class MeshTopo:
    """Mesh topology assistant."""

    def __init__(self, mesh: Mesh):
        self._mesh: Mesh = mesh

        # Cache members - will be computed on first access
        # Internal/Boundary flags
        self._internal_nodes: np.ndarray = None
        self._boundary_nodes: np.ndarray = None
        self._internal_faces: np.ndarray = None
        self._boundary_faces: np.ndarray = None
        self._internal_cells: np.ndarray = None
        self._boundary_cells: np.ndarray = None

        # Topology relations
        self._cell_neighbours: List[np.ndarray] = None
        self._node_neighbours: List[np.ndarray] = None
        self._face_cells: List[int] = None
        self._face_nodes: List[np.ndarray] = None
        self._node_faces: List[np.ndarray] = None
        self._node_cells: List[np.ndarray] = None
        self._cell_nodes: List[np.ndarray] = None
        self._cell_faces: List[np.ndarray] = None

    def reset(self, mesh: Mesh):
        """Resets all the cached topologies."""
        self.__init__(mesh)

    # -----------------------------------------------
    # region Continous properties
    # -----------------------------------------------

    @property
    def internal_nodes(self) -> np.ndarray:
        """Internal node ids."""
        if self._internal_nodes is None:
            self._calculate_flags()
        return self._internal_nodes

    @property
    def boundary_nodes(self) -> np.ndarray:
        """Boundary node ids."""
        if self._boundary_nodes is None:
            self._calculate_flags()
        return self._boundary_nodes

    @property
    def internal_faces(self) -> np.ndarray:
        """Internal face ids."""
        if self._internal_faces is None:
            self._calculate_flags()
        return self._internal_faces

    @property
    def boundary_faces(self) -> np.ndarray:
        """Boundary face ids."""
        if self._boundary_faces is None:
            self._calculate_flags()
        return self._boundary_faces

    @property
    def internal_cells(self) -> np.ndarray:
        """Internal cell ids."""
        if self._internal_cells is None:
            self._calculate_flags()
        return self._internal_cells

    @property
    def boundary_cells(self) -> np.ndarray:
        """Boundary cell ids."""
        if self._boundary_cells is None:
            self._calculate_flags()
        return self._boundary_cells

    def _calculate_flags(self):
        """Calculate and cache internal/boundary flags."""
        # Assume all as internal (True flag, False for boundary)
        node_flags = np.ones(self._mesh.node_count, dtype=bool)
        face_flags = np.zeros(self._mesh.face_count, dtype=bool)
        cell_flags = np.ones(self._mesh.cell_count, dtype=bool)

        # Calculate face_cells
        self._face_cells = self._get_face_cells_list()
        for f, (c1, c2) in enumerate(self._face_cells):
            if c1 is not None and c2 is not None:
                face_flags[f] = True  # Internal face

        # Mark nodes and cells based on face information
        for f, (c1, c2) in enumerate(self._face_cells):
            if not face_flags[f]:  # Boundary face
                # Correct the node flags
                for n in self._mesh.faces[f].nodes:
                    node_flags[n] = False
                # Correct the cell flags
                if c1 is not None:
                    cell_flags[c1] = False
                if c2 is not None:
                    cell_flags[c2] = False

        # Cache the flags
        self._internal_nodes = np.where(node_flags)[0]
        self._boundary_nodes = np.where(~node_flags)[0]

        self._internal_faces = np.where(face_flags)[0]
        self._boundary_faces = np.where(~face_flags)[0]

        self._internal_cells = np.where(cell_flags)[0]
        self._boundary_cells = np.where(~cell_flags)[0]

    def _get_face_cells_list(self):
        """Get the face_cells list."""
        face_cells_list = [None] * self._mesh.face_count
        cell_faces_map = collections.defaultdict(list)
        for c, cell in enumerate(self._mesh.cells):
            for f in cell.faces:
                cell_faces_map[f].append(c)

        for f, cells in cell_faces_map.items():
            if len(cells) == 1:
                # Boundary face: left cell as owner.
                face_cells_list[f] = [cells[0], None]
            elif len(cells) == 2:
                # Internal face
                cells = sorted(cells)  # c_left < c_right
                face_cells_list[f] = cells
            else:
                raise RuntimeError(f"Face {f} is shared by more than 2 cells: {cells}")

        return face_cells_list

    # -----------------------------------------------
    # region non-Continous attrs
    # -----------------------------------------------

    @property
    def face_nodes(self) -> List[np.ndarray]:
        """Face nodes in anticlockwise order."""
        if self._face_nodes is None:
            face_nodes_list = []
            for face in self._mesh.faces:
                node_ids = face.nodes
                nodes = self._mesh.get_nodes(node_ids)
                # Sort the nodes for 2D meshes
                nodes, node_ids = sort_anticlockwise(nodes, node_ids)
                face_nodes_list.append(np.array(node_ids, dtype=np.int32))
            self._face_nodes = face_nodes_list
        return self._face_nodes

    @property
    def face_cells(self) -> List[int]:
        """Face cells list."""
        if self._face_cells is None:
            self._calculate_flags()
        return self._face_cells

    @property
    def node_faces(self) -> List[np.ndarray]:
        """Node faces list."""
        if self._node_faces is None:
            node_faces_dict = collections.defaultdict(set)
            for f, face in enumerate(self._mesh.faces):
                for n in face.nodes:
                    node_faces_dict[n].add(f)
            if len(node_faces_dict) != self._mesh.node_count:
                raise RuntimeError("Some nodes are not connected to any face.")

            self._node_faces = [
                np.array(sorted(list(node_faces_dict[n])), dtype=np.int32)
                for n in range(self._mesh.node_count)
            ]
        return self._node_faces

    @property
    def node_cells(self) -> List[np.ndarray]:
        """Node cells list."""
        if self._node_cells is None:
            node_cells_dict = collections.defaultdict(set)
            for c, cell in enumerate(self._mesh.cells):
                for f in cell.faces:
                    for n in self._mesh.faces[f].nodes:
                        node_cells_dict[n].add(c)
            if len(node_cells_dict) != self._mesh.node_count:
                raise RuntimeError("Some nodes are not connected to any cell.")

            self._node_cells = [
                np.array(sorted(list(node_cells_dict[n])), dtype=np.int32)
                for n in range(self._mesh.node_count)
            ]

        return self._node_cells

    @property
    def cell_faces(self) -> List[np.ndarray]:
        """Cell faces sorted for 2d mesh."""
        if self._cell_faces is None:
            cell_faces_list = []
            for cell in self._mesh.cells:
                face_ids = cell.faces
                faces = self._mesh.get_faces(face_ids)
                if self._mesh.dimension == MeshDimension.D2:
                    _, face_ids = sort_anticlockwise(faces, face_ids)
                cell_faces_list.append(np.array(face_ids, dtype=np.int32))
            self._cell_faces = cell_faces_list
        return self._cell_faces

    @property
    def cell_nodes(self) -> List[np.ndarray]:
        """Cell nodes list."""
        if self._cell_nodes is None:
            cell_nodes_list = []
            for cell in self._mesh.cells:
                unique_nodes = set()
                for f in cell.faces:
                    unique_nodes.update(self._mesh.faces[f].nodes)
                cell_nodes_list.append(np.array(list(unique_nodes), dtype=np.int32))
            if len(cell_nodes_list) != self._mesh.cell_count:
                raise RuntimeError("Some cells are not connected to any node.")

            self._cell_nodes = cell_nodes_list
        return self._cell_nodes

    @property
    def cell_neighbours(self) -> List[np.ndarray]:
        """Cell neighbours list."""
        if self._cell_neighbours is None:
            num_cells = len(self._mesh.cells)
            neighbours_list = [set() for _ in range(num_cells)]

            for _, (c1, c2) in enumerate(self.face_cells):
                if c1 is not None and c2 is not None:
                    neighbours_list[c1].add(c2)
                    neighbours_list[c2].add(c1)
            if len(neighbours_list) != num_cells:
                raise RuntimeError("Some cells not connected to any others.")

            self._cell_neighbours = [
                np.array(list(nbrs), dtype=np.int32) for nbrs in neighbours_list
            ]
        return self._cell_neighbours

    @property
    def node_neighbours(self) -> List[np.ndarray]:
        """Node neighbours list."""
        if self._node_neighbours is None:
            node_faces_list = self.node_faces
            num_nodes = self._mesh.node_count
            node_neighbours_list = [
                np.array([], dtype=np.int32) for _ in range(num_nodes)
            ]

            for n in range(num_nodes):
                neighbours_set = set()
                for f in node_faces_list[n]:
                    for nbr_n in self._mesh.faces[f].nodes:
                        if nbr_n != n:
                            neighbours_set.add(nbr_n)
                node_neighbours_list[n] = np.array(
                    list(neighbours_set),
                    dtype=np.int32,
                )
            if len(node_neighbours_list) != num_nodes:
                raise RuntimeError("Some nodes not connected to any others.")

            self._node_neighbours = node_neighbours_list
        return self._node_neighbours
