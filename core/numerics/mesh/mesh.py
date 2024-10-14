# -*- encoding: utf-8 -*-
"""
Copyright (C) 2024, The YunmengEnvs Contributors. Join us, for you talents!  

Abstract mesh class for describing the geometry and topology.
"""
from core.numerics.mesh import Node, Face, Cell, Coordinate
from core.numerics.fields import Vector

from abc import ABC, abstractmethod
import numpy as np


class Mesh(ABC):
    """Abstract mesh class for describing the topology."""

    # -----------------------------------------------
    # --- properties ---
    # -----------------------------------------------

    @property
    @abstractmethod
    def version(self) -> int:
        """Return the version of the mesh."""
        pass

    @property
    @abstractmethod
    def domain(self) -> int:
        """Return the domain of the mesh, e.g. 1D, 2D or 3D."""
        pass

    @property
    @abstractmethod
    def extent(self) -> list:
        """Return the extent of the mesh."""
        pass

    @property
    @abstractmethod
    def node_count(self) -> int:
        """Return the number of nodes in the mesh."""
        pass

    @property
    @abstractmethod
    def face_count(self) -> int:
        """Return the number of faces in the mesh."""
        pass

    @property
    @abstractmethod
    def cell_count(self) -> int:
        """Return the number of cells in the mesh."""
        pass

    @property
    @abstractmethod
    def nodes(self) -> list:
        """Return all nodes in the mesh."""
        pass

    @property
    @abstractmethod
    def faces(self) -> list:
        """Return all faces in the mesh."""
        pass

    @property
    @abstractmethod
    def cells(self) -> list:
        """Return all cells in the mesh."""
        pass

    @property
    def info_node_elevation(self) -> tuple:
        """Return the min/max/avg elevation of nodes."""
        pass

    @property
    def info_face_area(self) -> tuple:
        """Return the min/max/avg area of faces."""
        pass

    @property
    def info_cell_volume(self) -> tuple:
        """Return the min/max/avg volume of cells."""
        pass

    # -----------------------------------------------
    # --- element access methods ---
    # -----------------------------------------------

    @abstractmethod
    def get_node(self, index: int) -> Node:
        """Return the node at the given index."""
        pass

    @abstractmethod
    def set_node(self, index: int, node: Node):
        """Set the node at the given index."""
        pass

    @abstractmethod
    def get_face(self, index: int) -> Face:
        """Return the face at the given index."""
        pass

    @abstractmethod
    def get_cell(self, index: int) -> Cell:
        """Return the cell at the given index."""
        pass

    @abstractmethod
    def refine_cell(self, cell_index: int, refine_level: int):
        """Refine the given cell by the given level."""
        pass

    @abstractmethod
    def relax_cell(self, cell_index: int):
        """Relax the given cell one level."""
        pass

    # -----------------------------------------------
    # --- additional methods ---
    # -----------------------------------------------

    @abstractmethod
    def set_node_group(self, group_name: str, node_indices: list):
        """Assign the given nodes to the given group."""
        pass

    @abstractmethod
    def set_face_group(self, group_name: str, face_indices: list):
        """Assign the given faces to the given group."""
        pass

    @abstractmethod
    def set_cell_group(self, group_name: str, cell_indices: list):
        """Assign the given cells to the given group."""
        pass

    @abstractmethod
    def get_group(self, group_name: str) -> list:
        """Return the element indices of given group."""
        pass

    @abstractmethod
    def get_boundary_face_indices(self) -> list:
        """Return a list of indices of boundary faces."""
        pass

    @abstractmethod
    def get_boundary_node_indices(self) -> list:
        """Return a list of indices of boundary nodes."""
        pass

    @abstractmethod
    def get_boundary_cell_indices(self) -> list:
        """Return a list of indices of boundary cells."""
        pass


class MeshTopo:
    """Mesh topology class for describing the topology.

    NOTE: require all faces to be continuously encoded.
    """

    def __init__(self, mesh: Mesh):
        self._mesh = mesh
        self._face_cells = None
        self._node_faces = None
        self._node_cells = None
        self._cell_nodes = None
        self._cell_neighbours = None

    def get_mesh(self) -> Mesh:
        """Return the bounded mesh."""
        return self._mesh

    # -----------------------------------------------
    # --- connectivity methods ---
    # -----------------------------------------------

    def collect_face_cells(self, face_index: int) -> list:
        """Collect the cells connected to the given face."""
        if self._face_cells is None:

            face_cells = [[] for _ in range(self._mesh.face_count)]
            for cell in self._mesh.cells:
                for face in cell.faces:
                    face_cells[face.id].append(cell.id)
            face_cells = [list(set(cells)) for cells in face_cells]
            self._face_cells = face_cells
        return self._face_cells[face_index]

    def collect_node_faces(self, node_index: int) -> list:
        """Collect the faces connected to the given node."""
        if self._node_faces is None:
            node_faces = [[] for _ in range(self._mesh.node_count)]
            for face in self._mesh.faces:
                for node in face.nodes:
                    node_faces[node.id].append(face.id)
            node_faces = [list(set(faces)) for faces in node_faces]
            self._node_faces = node_faces
        return self._node_faces[node_index]

    def collect_node_cells(self, node_index: int) -> list:
        """Collect the cells connected to the given node."""
        if self._node_cells is None:
            node_cells = [[] for _ in range(self._mesh.node_count)]
            for cell in self._mesh.cells:
                for face in cell.faces:
                    for node in face.nodes:
                        node_cells[node.id].append(cell.id)
            node_cells = [list(set(cells)) for cells in node_cells]
            self._node_cells = node_cells
        return self._node_cells[node_index]

    def collect_cell_nodes(self, cell_index: int) -> list:
        """Collect the nodes connected to the given cell."""
        if self._cell_nodes is None:
            cell_nodes = [[] for _ in range(self._mesh.cell_count)]
            for cell in self._mesh.cells:
                for face in cell.faces:
                    for node in face.nodes:
                        cell_nodes[cell.id].append(node.id)
            cell_nodes = [list(set(nodes)) for nodes in cell_nodes]
            self._cell_nodes = cell_nodes
        return self._cell_nodes[cell_index]

    def collect_cell_neighbours(self, cell_index: int) -> list:
        """Collect the neighbours of given cell."""
        if self._cell_neighbours is None:
            cell_neighbours = [[] for _ in range(self._mesh.cell_count)]
            for i in range(self._mesh.face_count):
                cells = self.collect_face_cells(i)
                if len(cells) == 2:
                    cell_neighbours[cells[0]].append(cells[1])
                    cell_neighbours[cells[1]].append(cells[0])
            self._cell_neighbours = cell_neighbours
        return self._cell_neighbours[cell_index]

    # -----------------------------------------------
    # --- retrieval methods ---
    # -----------------------------------------------

    def search_nearest_nodes(
        self, coord: Coordinate, max_dist: float, top_k: int
    ) -> list:
        """Search the k nearest nodes to the given coordinate within the given distance."""
        distances = []
        for node in self._mesh.nodes:
            dist = np.linalg.norm(node.coord.to_np() - coord.to_np())
            if dist <= max_dist:
                distances.append(node.id)
        distances.sort(key=lambda x: x)
        return distances[:top_k]

    def search_nearest_cell(self, coord: Coordinate, max_dist: float) -> int:
        """Search the nearest cell to the given coordinate within the given distance."""
        min_dist = float("inf")
        min_cell_index = -1
        for cell in self._mesh.cells:
            dist = np.linalg.norm(cell.center.to_np() - coord.to_np())
            if dist <= max_dist and dist < min_dist:
                min_dist = dist
                min_cell_index = cell.id
        return min_cell_index

    def search_nearest_face(self, coord: Coordinate, max_dist: float) -> int:
        """Search the nearest face to the given coordinate within the given distance."""
        min_dist = float("inf")
        min_face_index = -1
        for face in self._mesh.faces:
            dist = np.linalg.norm(face.center.to_np() - coord.to_np())
            if dist <= max_dist and dist < min_dist:
                min_dist = dist
                min_face_index = face.id
        return min_face_index

    def search_zone_cells(self, zone: list) -> list:
        """Search the cells within the given zone."""
        raise NotImplementedError


class MeshGeom:
    """Abstract mesh geometry class for describing the geometry."""

    def __init__(self, mesh: Mesh):
        self._mesh = mesh
        self._mesh_stats = None
        self._cell_to_cell_dists = None
        self._cell_to_face_dists = None
        self._cell_to_node_dists = None
        self._cell_to_cell_vects = None

    def get_mesh(self) -> Mesh:
        """Return the bounded mesh."""
        return self._mesh

    def calculate_cell_to_cell_distance(self, cell1: int, cell2: int) -> float:
        """Calculate the distance between the centroids of the given cells."""
        if self._cell_to_cell_dists is None:
            topos = MeshTopo(self._mesh)

            cell_num = self._mesh.cell_count
            cell_dists = [{} for _ in range(cell_num)]
            for i in range(cell_num):
                for j in topos.collect_cell_neighbours(i):
                    dist = np.linalg.norm(
                        self._mesh.cells[i].center.to_np()
                        - self._mesh.cells[j].center.to_np()
                    )
                    cell_dists[i][j] = dist
                    cell_dists[j][i] = dist
            self._cell_to_cell_dists = cell_dists

        if cell1 in self._cell_to_cell_dists:
            if cell2 in self._cell_to_cell_dists[cell1]:
                return self._cell_to_cell_dists[cell1][cell2]
        else:
            return None

    def calculate_cell_to_face_distance(self, cell: int, face: int) -> float:
        """Calculate the distance between the centroid of the given cell and the given face."""
        if self._cell_to_face_dists is None:
            cell_num = self._mesh.cell_count
            # NOTE: require all cells to be continuously encoded.
            cell_face_dists = [[] for _ in range(cell_num)]
            for cell in self._mesh.cells:
                for face in cell.faces:
                    dist = np.linalg.norm(cell.center.to_np() - face.center.to_np())
                    cell_face_dists[cell.id].append(dist)
            self._cell_to_face_dists = cell_face_dists
        return self._cell_to_face_dists[cell][face]

    def calculate_cell_to_node_distance(self, cell: int, node: int) -> float:
        """Calculate the distance between the given cell and the given node."""
        raise NotImplementedError

    def calculate_cell_to_cell_vector(self, cell1: int, cell2: int) -> Coordinate:
        """Calculate the unit vector from the given cells.

        NOTE: This method is not symmetric.
        """
        if self._cell_to_cell_vects is None:
            topos = MeshTopo(self._mesh)

            cell_num = self._mesh.cell_count
            cell_vecs = [{} for _ in range(cell_num)]
            for i in range(cell_num):
                for j in topos.collect_cell_neighbours(i):
                    vec_np = (
                        self._mesh.cells[i].center - self._mesh.cells[j].center
                    ).to_np()
                    vec = Vector().from_np(vec_np).unit()
                    cell_vecs[i][j] = vec
                    cell_vecs[j][i] = -vec
            self._cell_to_cell_vects = cell_vecs
        return self._cell_to_cell_vects[cell1][cell2]
