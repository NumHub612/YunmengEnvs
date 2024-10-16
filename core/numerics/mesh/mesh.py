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
    """Abstract mesh class for describing the topology.

    NOTE:
        - Don't support isolated element, no check yet.
    """

    # -----------------------------------------------
    # --- properties ---
    # -----------------------------------------------

    @property
    @abstractmethod
    def version(self) -> int:
        """Return the version."""
        pass

    @property
    @abstractmethod
    def domain(self) -> int:
        """Return the domain of the mesh, e.g. 1d, 2d or 3d."""
        pass

    @property
    @abstractmethod
    def extents(self) -> list:
        """Return the extents of each dimension."""
        pass

    @property
    @abstractmethod
    def node_count(self) -> int:
        """Return the number of nodes."""
        pass

    @property
    @abstractmethod
    def nodes(self) -> list:
        """Return all nodes."""
        pass

    @property
    def stat_node_elevation(self) -> tuple:
        """Return the min/max/avg elevation of nodes."""
        pass

    @property
    @abstractmethod
    def face_count(self) -> int:
        """Return the number of faces."""
        pass

    @property
    @abstractmethod
    def faces(self) -> list:
        """Return all faces."""
        pass

    @property
    def stat_face_area(self) -> tuple:
        """Return the min/max/avg area of faces."""
        pass

    @property
    @abstractmethod
    def cell_count(self) -> int:
        """Return the number of cells."""
        pass

    @property
    @abstractmethod
    def cells(self) -> list:
        """Return all cells."""
        pass

    @property
    def stat_cell_volume(self) -> tuple:
        """Return the min/max/avg volume of cells."""
        pass

    # -----------------------------------------------
    # --- element modification methods ---
    # -----------------------------------------------

    @abstractmethod
    def refine_cell(self, index: int):
        """Refine the given cell."""
        pass

    @abstractmethod
    def relax_cell(self, index: int):
        """Relax the given cell."""
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
    def delete_node_group(self, group_name: str):
        """Delete the given node group."""
        pass

    @abstractmethod
    def get_group(self, group_name: str) -> list:
        """Return the element indices of given group."""
        pass


class MeshTopo:
    """Mesh topology class for describing the topology.

    NOTE:
        - Require all elements to be continuously encoded, no check yet.
    """

    def __init__(self, mesh: Mesh):
        self._mesh = mesh
        self._boundary_faces = None
        self._interior_faces = None
        self._boundary_cells = None
        self._interior_cells = None
        self._boundary_nodes = None
        self._interior_nodes = None

        self._face_cells = None
        self._node_faces = None
        self._node_cells = None
        self._cell_nodes = None
        self._cell_neighbours = None
        self._node_neighbours = None

    def get_mesh(self) -> Mesh:
        """Return the bounded mesh."""
        return self._mesh

    # -----------------------------------------------
    # --- properties methods ---
    # -----------------------------------------------

    @property
    def boundary_faces_indexes(self) -> list:
        """Return the indexes of boundary faces."""
        if self._boundary_faces is None:
            self._boundary_faces = []
            for face in self._mesh.faces:
                if len(face.cells) == 1:
                    self._boundary_faces.append(face.id)
        return self._boundary_faces

    @property
    def interior_faces_indexes(self) -> list:
        """Return the indexes of interior faces."""
        if self._interior_faces is None:
            self._interior_faces = []
            for face in self._mesh.faces:
                if len(face.cells) == 2:
                    self._interior_faces.append(face.id)
        return self._interior_faces

    @property
    def boundary_cells_indexes(self) -> list:
        """Return the indexes of boundary cells."""
        if self._boundary_cells is None:
            self._boundary_cells = []
            for face in self.boundary_faces_indexes:
                cells = self.collect_face_cells(face)
                self._boundary_cells.extend(cells)
        return self._boundary_cells

    @property
    def interior_cells_indexes(self) -> list:
        """Return the indexes of interior cells."""
        if self._interior_cells is None:
            self._interior_cells = []
            for cell in self._mesh.cells:
                if cell.id not in self.boundary_cells_indexes:
                    self._interior_cells.append(cell.id)
        return self._interior_cells

    @property
    def boundary_nodes_indexes(self) -> list:
        """Return the indexes of boundary nodes."""
        if self._boundary_nodes is None:
            self._boundary_nodes = []
            for face in self.boundary_faces_indexes:
                nodes = self.collect_face_nodes(face)
                self._boundary_nodes.extend(nodes)
        return self._boundary_nodes

    @property
    def interior_nodes_indexes(self) -> list:
        """Return the indexes of interior nodes."""
        if self._interior_nodes is None:
            self._interior_nodes = []
            for node in self._mesh.nodes:
                if node.id not in self.boundary_nodes_indexes:
                    self._interior_nodes.append(node.id)
        return self._interior_nodes

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

    def collect_node_neighbours(self, node_index: int) -> list:
        """Collect the neighbours of given node."""
        if self._node_neighbours is None:
            node_neighbours = [[] for _ in range(self._mesh.node_count)]
            for face in self._mesh.faces:
                nodes = face.nodes
                for i in range(len(nodes)):
                    for j in range(i + 1, len(nodes)):
                        node_neighbours[nodes[i].id].append(nodes[j].id)
                        node_neighbours[nodes[j].id].append(nodes[i].id)
            node_neighbours = [list(set(nodes)) for nodes in node_neighbours]
            self._node_neighbours = node_neighbours
        return self._node_neighbours[node_index]

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

    # -----------------------------------------------
    # --- generation methods ---
    # -----------------------------------------------

    def generate_projection(self, coord: Coordinate, face_index: int) -> Coordinate:
        """Generate the projection of the given coordinate on the given face."""
        face = self._mesh.faces[face_index]
        normal = face.normal
        vec_np = (coord - face.center).to_np()
        proj_np = np.dot(vec_np, normal.to_np()) * normal.to_np()
        proj_coord = Coordinate.from_np(proj_np + face.center.to_np())
        return proj_coord


class MeshGeom:
    """Mesh geometry class for calculating the geometry.

    NOTE:
        - Require all elements to be continuously encoded, no check yet.
        - Return None if the elements have no connection.
    """

    def __init__(self, mesh: Mesh):
        self._mesh = mesh

        self._cell_to_cell_dists = None
        self._cell_to_face_dists = None
        self._cell_to_node_dists = None
        self._node_to_node_dists = None

        self._cell_to_cell_vects = None
        self._cell_to_face_vects = None

    def get_mesh(self) -> Mesh:
        """Return the bounded mesh."""
        return self._mesh

    # -----------------------------------------------
    # --- geometry methods ---
    # -----------------------------------------------

    def calculate_cell_to_cell_distance(self, cell1: int, cell2: int) -> float:
        """Calculate the distance between the centroids of the given cells."""
        if self._cell_to_cell_dists is None:
            cell_num = self._mesh.cell_count
            cell_dists = [{} for _ in range(cell_num)]

            topos = MeshTopo(self._mesh)
            for i in range(cell_num):
                for j in topos.collect_cell_neighbours(i):
                    dist = np.linalg.norm(
                        self._mesh.cells[i].center.to_np()
                        - self._mesh.cells[j].center.to_np()
                    )
                    cell_dists[i][j] = dist
                    cell_dists[j][i] = dist

            self._cell_to_cell_dists = cell_dists

        if cell2 in self._cell_to_cell_dists[cell1]:
            return self._cell_to_cell_dists[cell1][cell2]
        else:
            return None

    def calculate_cell_to_face_distance(self, cell: int, face: int) -> float:
        """Calculate the distance between the centroid of the given cell and the given face."""
        if self._cell_to_face_dists is None:
            cell_num = self._mesh.cell_count
            cell_face_dists = [{} for _ in range(cell_num)]
            for cell in self._mesh.cells:
                for face in cell.faces:
                    dist = np.linalg.norm(cell.center.to_np() - face.center.to_np())
                    cell_face_dists[cell.id][face.id] = dist

            self._cell_to_face_dists = cell_face_dists

        if face in self._cell_to_face_dists[cell]:
            return self._cell_to_face_dists[cell][face]
        else:
            return None

    def calculate_cell_to_node_distance(self, cell: int, node: int) -> float:
        """Calculate the distance between the given cell and the given node."""
        if self._cell_to_node_dists is None:
            cell_num = self._mesh.cell_count
            cell_node_dists = [{} for _ in range(cell_num)]

            topos = MeshTopo(self._mesh)
            for i in range(cell_num):
                for j in topos.collect_cell_nodes(i):
                    dist = np.linalg.norm(
                        self._mesh.cells[i].center.to_np()
                        - self._mesh.nodes[j].coord.to_np()
                    )
                    cell_node_dists[i][j] = dist

            self._cell_to_node_dists = cell_node_dists

        if node in self._cell_to_node_dists[cell]:
            return self._cell_to_node_dists[cell][node]
        else:
            return None

    def calucate_node_to_node_distance(self, node1: int, node2: int) -> float:
        """Calculate the distance between the given nodes."""
        if self._node_to_node_dists is None:
            node_num = self._mesh.node_count
            node_dists = [{} for _ in range(node_num)]

            topos = MeshTopo(self._mesh)
            for i in range(node_num):
                for j in topos.collect_node_neighbours(i):
                    dist = np.linalg.norm(
                        self._mesh.nodes[i].coord.to_np()
                        - self._mesh.nodes[j].coord.to_np()
                    )
                    node_dists[i][j] = dist
                    node_dists[j][i] = dist

            self._node_to_node_dists = node_dists

        if node2 in self._node_to_node_dists[node1]:
            return self._node_to_node_dists[node1][node2]
        else:
            return None

    # -----------------------------------------------
    # --- vector methods ---
    # -----------------------------------------------

    def calculate_cell_to_cell_vector(self, cell1: int, cell2: int) -> Coordinate:
        """Calculate the unit vector from the given cells.

        NOTE:
            - This method is not symmetric.
        """
        if self._cell_to_cell_vects is None:
            cell_num = self._mesh.cell_count
            cell_vecs = [{} for _ in range(cell_num)]

            topos = MeshTopo(self._mesh)
            for i in range(cell_num):
                for j in topos.collect_cell_neighbours(i):
                    vec_np = (
                        self._mesh.cells[i].center - self._mesh.cells[j].center
                    ).to_np()
                    vec = Vector.from_np(vec_np).unit()
                    cell_vecs[i][j] = vec
                    cell_vecs[j][i] = -vec

            self._cell_to_cell_vects = cell_vecs

        if cell2 in self._cell_to_cell_vects[cell1]:
            return self._cell_to_cell_vects[cell1][cell2]
        else:
            return None

    def calculate_cell_to_face_vector(self, cell: int, face: int) -> Coordinate:
        """Calculate the unit vector from the given cell to the given face."""
        if self._cell_to_face_vects is None:
            cell_num = self._mesh.cell_count
            cell_face_vecs = [{} for _ in range(cell_num)]
            for cell in self._mesh.cells:
                for face in cell.faces:
                    vec_np = (cell.center - face.center).to_np()
                    vec = Vector.from_np(vec_np).unit()
                    cell_face_vecs[cell.id][face.id] = vec

            self._cell_to_face_vects = cell_face_vecs

        if face in self._cell_to_face_vects[cell]:
            return self._cell_to_face_vects[cell][face]
        else:
            return None
