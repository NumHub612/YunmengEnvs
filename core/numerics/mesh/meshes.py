# -*- encoding: utf-8 -*-
"""
Copyright (C) 2024, The YunmengEnvs Contributors. Join us, share your ideas!  

Abstract mesh class for describing the geometry and topology.
"""
from core.numerics.mesh import Coordinate, Node, Face, Cell
from core.numerics.fields import Vector
from configs.settings import logger

from abc import ABC, abstractmethod
import numpy as np
import math


class Mesh(ABC):
    """Abstract mesh class for describing the topology.

    Notes:
        - Don't support isolated element.
    """

    def __init__(self):
        self._version = 1
        self._nodes = []
        self._faces = []
        self._cells = []
        self._groups = {}

        self._topo = None
        self._geom = None

    # -----------------------------------------------
    # --- properties ---
    # -----------------------------------------------

    @property
    def node_indexes(self) -> list:
        """Return the indexes of all nodes."""
        return [node.id for node in self._nodes]

    @property
    def node_count(self) -> int:
        """Return the number of nodes."""
        return len(self._nodes)

    @property
    def nodes(self) -> list:
        """Return all nodes."""
        return self._nodes

    @property
    def face_indexes(self) -> list:
        """Return the indexes of all faces."""
        return [face.id for face in self._faces]

    @property
    def face_count(self) -> int:
        """Return the number of faces."""
        return len(self._faces)

    @property
    def faces(self) -> list:
        """Return all faces."""
        return self._faces

    @property
    def cell_indexes(self) -> list:
        """Return the indexes of all cells."""
        return [cell.id for cell in self._cells]

    @property
    def cell_count(self) -> int:
        """Return the number of cells."""
        return len(self._cells)

    @property
    def cells(self) -> list:
        """Return all cells."""
        return self._cells

    @property
    def version(self) -> int:
        """Return the version."""
        return self._version

    @property
    @abstractmethod
    def dimension(self) -> str:
        """Return the mesh domain, e.g. 1d, 2d or 3d."""
        pass

    # -----------------------------------------------
    # --- mesh modification methods ---
    # -----------------------------------------------

    @abstractmethod
    def refine_cells(self, indexes: list):
        """Refine the given cell."""
        pass

    @abstractmethod
    def relax_cells(self, indexes: list):
        """Relax the given cell."""
        pass

    # -----------------------------------------------
    # --- additional methods ---
    # -----------------------------------------------

    def set_node_group(self, group_name: str, node_indices: list):
        """Assign the given nodes to the given group."""
        if group_name in self._groups:
            raise ValueError(f"Group {group_name} already exists.")

        min_id, max_id, _ = self.get_geom_assistant().statistics_node_attribute("id")
        for node_index in node_indices:
            if node_index > max_id or node_index < min_id:
                raise ValueError(f"Node index {node_index} out of range.")

        self._groups[group_name] = node_indices

    def set_face_group(self, group_name: str, face_indices: list):
        """Assign the given faces to the given group."""
        if group_name in self._groups:
            raise ValueError(f"Group {group_name} already exists.")

        min_id, max_id, _ = self.get_geom_assistant().statistics_face_attribute("id")
        for face_index in face_indices:
            if face_index > max_id or face_index < min_id:
                raise ValueError(f"Face index {face_index} out of range.")

        self._groups[group_name] = face_indices

    def set_cell_group(self, group_name: str, cell_indices: list):
        """Assign the given cells to the given group."""
        if group_name in self._groups:
            raise ValueError(f"Group {group_name} already exists.")

        min_id, max_id, _ = self.get_geom_assistant().statistics_cell_attribute("id")
        for cell_index in cell_indices:
            if cell_index > max_id or cell_index < min_id:
                raise ValueError(f"Cell index {cell_index} out of range.")

        self._groups[group_name] = cell_indices

    def delete_group(self, group_name: str):
        """Delete the given node group."""
        if group_name in self._groups:
            self._groups.pop(group_name)

    def get_group(self, group_name: str) -> list:
        """Return the element indices of given group."""
        if group_name in self._groups:
            return self._groups[group_name]
        else:
            return []

    # -----------------------------------------------
    # --- extension methods ---
    # -----------------------------------------------

    def get_topo_assistant(self) -> "MeshTopo":
        """Return the mesh topology assistant."""
        if self._topo is None:
            self._topo = MeshTopo(self)
        return self._topo

    def get_geom_assistant(self) -> "MeshGeom":
        """Return the mesh geometry assistant."""
        if self._geom is None:
            self._geom = MeshGeom(self)
        return self._geom


class MeshTopo:
    """Mesh topology class for describing the topology.

    NOTE:
        - This class is lazily, i.e. it will not be initialized until it is needed.
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
    def boundary_nodes_indexes(self) -> list:
        """Return the indexes of boundary nodes."""
        if self._boundary_nodes is None:
            self._boundary_nodes = []
            for fid in self.boundary_faces_indexes:
                self._boundary_nodes.extend(self._mesh.faces[fid].nodes)
            self._boundary_nodes = list(set(self._boundary_nodes))
        return self._boundary_nodes

    @property
    def interior_nodes_indexes(self) -> list:
        """Return the indexes of interior nodes."""
        if self._interior_nodes is None:
            self._interior_nodes = []
            for node in self._mesh.nodes:
                if node.id not in self.boundary_nodes_indexes:
                    self._interior_nodes.append(node.id)
            self._interior_nodes = list(set(self._interior_nodes))
        return self._interior_nodes

    @property
    def boundary_faces_indexes(self) -> list:
        """Return the indexes of boundary faces."""
        if self._boundary_faces is None:
            self._boundary_faces = []
            for face in self._mesh.faces:
                cells = self.collect_face_cells(face.id)
                if len(cells) == 1:
                    self._boundary_faces.append(face.id)
        return self._boundary_faces

    @property
    def interior_faces_indexes(self) -> list:
        """Return the indexes of interior faces."""
        if self._interior_faces is None:
            self._interior_faces = []
            for face in self._mesh.faces:
                if face.id not in self.boundary_faces_indexes:
                    self._interior_faces.append(face.id)
        return self._interior_faces

    @property
    def boundary_cells_indexes(self) -> list:
        """Return the indexes of boundary cells."""
        if self._boundary_cells is None:
            self._boundary_cells = []
            for fid in self.boundary_faces_indexes:
                cids = self.collect_face_cells(fid)
                self._boundary_cells.extend(cids)
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

    # -----------------------------------------------
    # --- connectivity methods ---
    # -----------------------------------------------

    def collect_node_neighbours(self, node_index: int) -> list:
        """Collect the neighbours indexes of given node."""
        if self._node_neighbours is None:
            neighbours = {nid: [] for nid in self._mesh.node_indexes}
            if self._mesh.dimension == "1d":
                print("Warning: 1D mesh neighbour search is going to be abandoned.")
                for cell in self._mesh.cells:
                    faces = cell.faces
                    for i in range(len(faces)):
                        nid1 = self._mesh.faces[faces[i]].nodes[0]
                        for j in range(i + 1, len(faces)):
                            nid2 = self._mesh.faces[faces[j]].nodes[0]
                            neighbours[nid1].append(nid2)
                            neighbours[nid2].append(nid1)
            else:
                for face in self._mesh.faces:
                    nodes = face.nodes
                    for i in range(len(nodes)):
                        for j in range(i + 1, len(nodes)):
                            neighbours[nodes[i]].append(nodes[j])
                            neighbours[nodes[j]].append(nodes[i])
            neighbours = {nid: list(set(nodes)) for nid, nodes in neighbours.items()}
            self._node_neighbours = neighbours
        return self._node_neighbours[node_index]

    def collect_face_cells(self, face_index: int) -> list:
        """Collect the cells connected to the given face."""
        if self._face_cells is None:
            face_cells = {fid: [] for fid in self._mesh.face_indexes}
            for cell in self._mesh.cells:
                for fid in cell.faces:
                    face_cells[fid].append(cell.id)
            face_cells = {fid: list(set(cells)) for fid, cells in face_cells.items()}
            self._face_cells = face_cells
        return self._face_cells[face_index]

    def collect_node_faces(self, node_index: int) -> list:
        """Collect the faces connected to the given node."""
        if self._node_faces is None:
            node_faces = {nid: [] for nid in self._mesh.node_indexes}
            for face in self._mesh.faces:
                for nid in face.nodes:
                    node_faces[nid].append(face.id)
            node_faces = {nid: list(set(faces)) for nid, faces in node_faces.items()}
            self._node_faces = node_faces
        return self._node_faces[node_index]

    def collect_node_cells(self, node_index: int) -> list:
        """Collect the cells connected to the given node."""
        if self._node_cells is None:
            node_cells = {nid: [] for nid in self._mesh.node_indexes}
            for cell in self._mesh.cells:
                for fid in cell.faces:
                    for nid in self._mesh.faces[fid].nodes:
                        node_cells[nid].append(cell.id)
            node_cells = {nid: list(set(cells)) for nid, cells in node_cells.items()}
            self._node_cells = node_cells
        return self._node_cells[node_index]

    def collect_cell_nodes(self, cell_index: int) -> list:
        """Collect the nodes connected to the given cell."""
        if self._cell_nodes is None:
            cell_nodes = {cid: [] for cid in self._mesh.cell_indexes}
            for cell in self._mesh.cells:
                for fid in cell.faces:
                    for nid in self._mesh.faces[fid].nodes:
                        cell_nodes[cell.id].append(nid)
            cell_nodes = {cid: list(set(nodes)) for cid, nodes in cell_nodes.items()}
            self._cell_nodes = cell_nodes
        return self._cell_nodes[cell_index]

    def collect_cell_neighbours(self, cell_index: int) -> list:
        """Collect the neighbours of given cell."""
        if self._cell_neighbours is None:
            cell_neighbours = {cid: [] for cid in self._mesh.cell_indexes}
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
        self,
        coordinate: Coordinate,
        max_dist: float,
        top_k: int,
    ) -> list:
        """Search the k nearest nodes to the given coordinate."""
        distances = []
        for node in self._mesh.nodes:
            dist = np.linalg.norm(node.coordinate.to_np() - coordinate.to_np())
            if dist <= max_dist:
                distances.append(node.id)
        distances.sort(key=lambda x: x)
        return distances[:top_k]

    def search_nearest_cell(
        self,
        coordinate: Coordinate,
        max_dist: float,
    ) -> int:
        """Search the nearest cell to the given coordinate."""
        min_dist = float("inf")
        min_cell_index = -1
        for cell in self._mesh.cells:
            dist = np.linalg.norm(cell.coordinate.to_np() - coordinate.to_np())
            if dist <= max_dist and dist < min_dist:
                min_dist = dist
                min_cell_index = cell.id
        return min_cell_index

    def search_nearest_face(
        self,
        coordinate: Coordinate,
        max_dist: float,
    ) -> int:
        """Search the nearest face to the given coordinate."""
        min_dist = float("inf")
        min_face_index = -1
        for face in self._mesh.faces:
            dist = np.linalg.norm(face.coordinate.to_np() - coordinate.to_np())
            if dist <= max_dist and dist < min_dist:
                min_dist = dist
                min_face_index = face.id
        return min_face_index

    # -----------------------------------------------
    # --- generation methods ---
    # -----------------------------------------------

    def generate_projection(
        self,
        coordinate: Coordinate,
        face_index: int,
    ) -> Coordinate:
        """Generate the projection of the given coordinate on the given face."""
        face = self._mesh.faces[face_index]
        normal = face.normal
        vec_np = (coordinate - face.coordinate).to_np()
        proj_np = np.dot(vec_np, normal.to_np()) * normal.to_np()
        proj_coord = Coordinate.from_np(proj_np + face.coordinate.to_np())
        return proj_coord


class MeshGeom:
    """Mesh geometry class for calculating the geometry.

    NOTE:
        - This class is lazily, i.e. it will not be initialized until it is needed.
    """

    def __init__(self, mesh: Mesh):
        self._mesh = mesh

        self._cell_to_cell_dists = None
        self._cell_to_face_dists = None
        self._cell_to_node_dists = None
        self._node_to_node_dists = None

        self._cell_to_cell_vects = None
        self._cell_to_face_vects = None

        self._stats = {"node": {}, "face": {}, "cell": {}}

    def get_mesh(self) -> Mesh:
        """Return the bounded mesh."""
        return self._mesh

    # -----------------------------------------------
    # --- static methods ---
    # -----------------------------------------------

    @staticmethod
    def calculate_distance(coord1: Coordinate, coord2: Coordinate) -> float:
        """Calculate the distance between two coordinates."""
        return np.linalg.norm(coord1.to_np() - coord2.to_np())

    @staticmethod
    def calculate_center(coords: list) -> Coordinate:
        """Calculate the center of the given coordinates."""
        return Coordinate.from_np(np.mean([coord.to_np() for coord in coords], axis=0))

    @staticmethod
    def sort_anticlockwise(coords: dict, ignored_axis: str = "z") -> list:
        """Sort the coordinates in anticlockwise order.

        Args:
            coordinates: Coodinates with id as key.
            ignored_axis: The axis to be folded.

        Returns:
            Ids of sorted coordinates.
        """
        # Calculate the center of the coordinates
        center = MeshGeom.calculate_center(coords.values())

        # Sort the coordinates by their angle with the center
        ignored_axis = ignored_axis.lower()
        if ignored_axis == "z":
            sorted_coords = sorted(
                coords.items(),
                key=lambda x: math.atan2(x[1].y - center.y, x[1].x - center.x),
            )
        elif ignored_axis == "y":
            sorted_coords = sorted(
                coords.items(),
                key=lambda x: math.atan2(x[1].z - center.z, x[1].x - center.x),
            )
        elif ignored_axis == "x":
            sorted_coords = sorted(
                coords.items(),
                key=lambda x: math.atan2(x[1].y - center.y, x[1].z - center.z),
            )
        else:
            raise ValueError("Invalid ignored_axis: {}".format(ignored_axis))

        # Return the ids of the sorted coordinates
        return [coord_id for coord_id, _ in sorted_coords]

    @staticmethod
    def calculate_area(coords: list) -> float:
        """Calculate the area of the given coordinates.

        Note:
            The coordinates should be sorted.
        """
        if len(coords) < 3:
            raise ValueError("At least 3 coordinates are required.")

        # Calculate the center of the coordinates
        center = MeshGeom.calculate_center(coords)

        # Calculate the area using the shoelace formula
        area = 0.0
        for i in range(len(coords)):
            j = (i + 1) % len(coords)
            area += (coords[i].x - center.x) * (coords[j].y - center.y)
            area -= (coords[i].y - center.y) * (coords[j].x - center.x)
        area /= 2.0

        return abs(area)

    @staticmethod
    def calculate_perimeter(coords: list) -> float:
        """Calculate the perimeter of the given coordinates.

        Note:
            The coordinates should be sorted.
        """
        if len(coords) < 2:
            raise ValueError("At least 2 coordinates are required.")

        # Calculate the perimeter using the distance formula
        perimeter = 0.0
        for i in range(len(coords)):
            perimeter += MeshGeom.calculate_distance(
                coords[i], coords[(i + 1) % len(coords)]
            )
        return perimeter

    # -----------------------------------------------
    # --- statistics methods ---
    # -----------------------------------------------

    def statistics_node_attribute(self, attribute: str) -> tuple:
        """Statistics of the given attribute for all mesh nodes.

        Args:
            attribute: The attribute.

        Returns:
            A tuple of (min, max, avg) of the given attribute.

        Note:
            The attribute should be a valid attribute of Node:
            - x, y, z: float
            - id: int
        """
        attr = attribute.lower()
        if attr not in ["x", "y", "z", "id"]:
            raise ValueError(f"Invalid Node attribute: {attribute}")

        if attr in self._stats["node"]:
            return self._stats["node"][attr]

        if attr == "id":
            values = [node.id for node in self._mesh.nodes]
        else:
            values = [getattr(node.coordinate, attr) for node in self._mesh.nodes]

        self._stats["node"][attr] = (
            min(values),
            max(values),
            sum(values) / len(values),
        )
        return self._stats["node"][attr]

    def statistics_face_attribute(self, attribute: str) -> tuple:
        """Statistics of the given attribute for all mesh faces.

        Args:
            attribute: The attribute.

        Returns:
            A tuple of (min, max, avg) of the given attribute.

        Note:
            The attribute should be a valid attribute of Face:
            - x, y, z: float
            - area: float
            - perimeter: Vector
            - id: int
        """
        attr = attribute.lower()
        if attr not in ["x", "y", "z", "area", "perimeter", "id"]:
            raise ValueError(f"Invalid Face attribute: {attribute}")

        if attr in self._stats["face"]:
            return self._stats["face"][attr]

        if attr not in ["x", "y", "z"]:
            values = [getattr(face, attr) for face in self._mesh.faces]
        else:
            values = [getattr(face.coordinate, attr) for face in self._mesh.faces]

        self._stats["face"][attr] = (
            min(values),
            max(values),
            sum(values) / len(values),
        )
        return self._stats["face"][attr]

    def statistics_cell_attribute(self, attribute: str) -> tuple:
        """Statistics of the given attribute for all mesh cells.

        Args:
            attribute: The attribute.

        Returns:
            A tuple of (min, max, avg) of the given attribute.

        Note:
            The attribute should be a valid attribute of Cell:
            - x, y, z: float
            - volume: float
            - surface: float
            - id: int
        """
        attr = attribute.lower()
        if attr not in ["x", "y", "z", "volume", "surface", "id"]:
            raise ValueError(f"Invalid Cell attribute: {attribute}")

        if attr in self._stats["cell"]:
            return self._stats["cell"][attr]

        if attr not in ["x", "y", "z"]:
            values = [getattr(cell, attr) for cell in self._mesh.cells]
        else:
            values = [getattr(cell.coordinate, attr) for cell in self._mesh.cells]

        self._stats["cell"][attr] = (
            min(values),
            max(values),
            sum(values) / len(values),
        )
        return self._stats["cell"][attr]

    # -----------------------------------------------
    # --- geometry methods ---
    # -----------------------------------------------

    def calculate_cell_to_cell_distance(self, cell1: int, cell2: int) -> float:
        """Calculate the distance between the centroids of cells."""
        if self._cell_to_cell_dists is None:
            cell_dists = {cid: {} for cid in self._mesh.cell_indexes}
            topos = self._mesh.get_topo_assistant()
            for cid in self._mesh.cell_indexes:
                for j in topos.collect_cell_neighbours(cid):
                    dist = self.calculate_distance(
                        self._mesh.cells[cid].coordinate, self._mesh.cells[j].coordinate
                    )
                    cell_dists[cid][j] = dist
                    cell_dists[j][cid] = dist
            self._cell_to_cell_dists = cell_dists

        if cell2 in self._cell_to_cell_dists[cell1]:
            return self._cell_to_cell_dists[cell1][cell2]
        else:
            return None

    def calculate_cell_to_face_distance(self, cell: int, face: int) -> float:
        """Calculate the distance between the given cell and the given face."""
        if self._cell_to_face_dists is None:
            cell_face_dists = {cid: {} for cid in self._mesh.cell_indexes}
            for cell in self._mesh.cells:
                for fid in cell.faces:
                    dist = self.calculate_distance(
                        cell.coordinate, self._mesh.faces[fid].coordinate
                    )
                    cell_face_dists[cell.id][fid] = dist
            self._cell_to_face_dists = cell_face_dists

        if face in self._cell_to_face_dists[cell]:
            return self._cell_to_face_dists[cell][face]
        else:
            return None

    def calculate_cell_to_node_distance(self, cell: int, node: int) -> float:
        """Calculate the distance between the given cell and the given node."""
        if self._cell_to_node_dists is None:
            cell_node_dists = {cid: {} for cid in self._mesh.cell_indexes}
            topos = self._mesh.get_topo_assistant()
            for cid in self._mesh.cell_indexes:
                for j in topos.collect_cell_nodes(cid):
                    dist = self.calculate_distance(
                        self._mesh.cells[cid].coordinate, self._mesh.nodes[j].coordinate
                    )
                    cell_node_dists[cid][j] = dist
            self._cell_to_node_dists = cell_node_dists

        if node in self._cell_to_node_dists[cell]:
            return self._cell_to_node_dists[cell][node]
        else:
            return None

    def calucate_node_to_node_distance(self, node1: int, node2: int) -> float:
        """Calculate the distance between the given nodes."""
        if self._node_to_node_dists is None:
            node_dists = {nid: {} for nid in self._mesh.node_indexes}
            topos = self._mesh.get_topo_assistant()
            for nid in self._mesh.node_indexes:
                for j in topos.collect_node_neighbours(nid):
                    dist = self.calculate_distance(
                        self._mesh.nodes[nid].coordinate, self._mesh.nodes[j].coordinate
                    )
                    node_dists[nid][j] = dist
                    node_dists[j][nid] = dist
            self._node_to_node_dists = node_dists

        if node2 in self._node_to_node_dists[node1]:
            return self._node_to_node_dists[node1][node2]
        else:
            return None

    # -----------------------------------------------
    # --- generation methods ---
    # -----------------------------------------------

    def extract_coordinates_separated(
        self,
        element_type: str,
        dims: str = "xyz",
    ) -> list:
        """Extract the coordinates of each element separatedly."""
        etype = element_type.lower()
        if etype not in ["node", "cell", "face"]:
            raise ValueError(f"Invalid element type: {etype}.")

        dims = dims.lower()
        if dims not in ["xyz", "xy", "xz", "yz", "x", "y", "z"]:
            raise ValueError(f"Invalid dimension: {dims}.")

        elements_map = {
            "node": self._mesh.nodes,
            "cell": self._mesh.cells,
            "face": self._mesh.faces,
        }
        elements = elements_map.get(etype)

        xs = np.array([e.coordinate.x for e in elements])
        ys = np.array([e.coordinate.y for e in elements])
        zs = np.array([e.coordinate.z for e in elements])

        coordinate_map = {"x": xs, "y": ys, "z": zs}
        coordinates = [coordinate_map.get(d) for d in dims]
        return coordinates

    def extract_coordinates(self, element_type: str) -> np.ndarray:
        """Extract the coordinates of all elements."""
        etype = element_type.lower()
        if etype not in ["node", "cell", "face"]:
            raise ValueError(f"Invalid element type: {etype}.")

        elements_map = {
            "node": self._mesh.nodes,
            "cell": self._mesh.cells,
            "face": self._mesh.faces,
        }
        elements = elements_map.get(etype)

        coordinates = np.array([e.coordinate.to_np() for e in elements])
        return coordinates

    # -----------------------------------------------
    # --- vector methods ---
    # -----------------------------------------------

    def calculate_cell_to_cell_vector(self, cell1: int, cell2: int) -> Coordinate:
        """Calculate the unit vector from the given cells.

        NOTE:
            - This method is not symmetric.
        """
        if self._cell_to_cell_vects is None:
            cell_vecs = {cid: {} for cid in self._mesh.cell_indexes}
            topos = self._mesh.get_topo_assistant()
            for cid in self._mesh.cell_indexes:
                for j in topos.collect_cell_neighbours(cid):
                    vec_np = (
                        self._mesh.cells[cid].coordinate
                        - self._mesh.cells[j].coordinate
                    ).to_np()
                    vec = Vector.from_np(vec_np)
                    vec /= vec.magnitude
                    cell_vecs[cid][j] = vec
                    cell_vecs[j][cid] = -vec
            self._cell_to_cell_vects = cell_vecs

        if cell2 in self._cell_to_cell_vects[cell1]:
            return self._cell_to_cell_vects[cell1][cell2]
        else:
            return None

    def calculate_cell_to_face_vector(self, cell: int, face: int) -> Coordinate:
        """Calculate the unit vector from the given cell to the given face."""
        if self._cell_to_face_vects is None:
            cell_face_vecs = {cid: {} for cid in self._mesh.cell_indexes}
            for cell in self._mesh.cells:
                for face in cell.faces:
                    vec_np = (cell.coordinate - face.coordinate).to_np()
                    vec = Vector.from_np(vec_np)
                    vec /= vec.magnitude
                    cell_face_vecs[cell.id][face.id] = vec
            self._cell_to_face_vects = cell_face_vecs

        if face in self._cell_to_face_vects[cell]:
            return self._cell_to_face_vects[cell][face]
        else:
            return None


class GenericMesh(Mesh):
    """Generic mesh."""

    def __init__(self, nodes: list, faces: list, cells: list):
        """Generic mesh.

        Args:
            nodes: The list of nodes cooridnates.
            faces: The list of faces with their nodes indexes.
            cells: The list of cells with their faces indexes.

        Example:
            For a 2D unstructured mesh:
            >>> nodes = [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
            ]
            >>> faces = [
                [0, 1],
                [1, 2],
                [2, 3],
                [3, 0],
            ]
            >>> cells = [
                [0, 1, 2, 3],
            ]
            >>> mesh = GenericMesh(nodes, faces, cells)
        """
        super().__init__()
        self._dimension = ""

        self._generate_nodes(nodes)
        self._generate_faces(faces)
        self._generate_cells(cells)

        self._topo = MeshTopo(self)
        self._geom = MeshGeom(self)

    def _generate_nodes(self, nodes: list):
        """Generate the nodes of the mesh."""
        for i, coor in enumerate(nodes):
            self._nodes.append(Node(i, Coordinate(*coor)))

    def _generate_faces(self, faces: list):
        """Generate the faces of the mesh."""
        for i, node_ids in enumerate(faces):
            coors = [self._nodes[j].coordinate for j in node_ids]
            center = MeshGeom.calculate_center(coors)
            # For 2d mesh
            if len(node_ids) == 2:
                self._dimension = "2d"
                perimeter = MeshGeom.calculate_distance(coors[0], coors[1])
                area = perimeter
                normal = Vector.from_np(
                    np.cross(
                        (coors[1].to_np() - coors[0].to_np()),
                        [0, 0, 1],
                    )
                )
                normal /= normal.magnitude
            # For 3d mesh
            else:
                self._dimension = "3d"
                normal, perimeter, area = self._calculate_plane_features(coors)

            face = Face(i, center, node_ids, perimeter, area, normal)
            self._faces.append(face)

    def _generate_cells(self, cells: list):
        """Generate the cells of the mesh."""
        for i, face_ids in enumerate(cells):
            faces = [self._faces[j] for j in face_ids]
            center = MeshGeom.calculate_center([face.coordinate for face in faces])
            surface = sum([face.area for face in faces])
            volume = 0.0
            if self._dimension == "2d":
                node_coors = {
                    node_id: self._nodes[node_id].coordinate
                    for face in faces
                    for node_id in face.nodes
                }
                _, _, area = self._calculate_plane_features(list(node_coors.values()))
                volume = area

                coors = {face.id: face.coordinate for face in faces}
                _, dir_axis = self._calculate_plane_normal(list(coors.values()))
                face_ids = MeshGeom.sort_anticlockwise(coors, dir_axis)
            else:
                # For tetrahedron
                if len(faces) == 4:
                    node_coors = {
                        node_id: self._nodes[node_id].coordinate
                        for face in faces
                        for node_id in face.nodes
                    }
                    volume = self._calcuate_tetrahedron_volume(
                        list(node_coors.values())
                    )
                # For hexahedron
                elif len(faces) == 8:
                    coors = {face.id: face.coordinate for face in faces}
                    volume = self._calculate_hexahedron_volume(coors, center)
                else:
                    raise ValueError("Unsupported cell")

            cell = Cell(i, center, face_ids, surface, volume)
            self._cells.append(cell)

    def _calculate_plane_normal(self, coors: list) -> tuple:
        """Calculate the normal of the plane"""
        if len(coors) < 3:
            raise ValueError("At least 3 coordinates are required.")

        # Calculate the normal using the shoelace formula
        normal = Vector.from_np(
            np.cross(
                (coors[1].to_np() - coors[0].to_np()),
                (coors[2].to_np() - coors[1].to_np()),
            )
        )
        normal /= normal.magnitude

        # Check the direction of the normal
        dir_index = np.argmax(np.abs(normal.to_np()))
        dir_map = {0: "x", 1: "y", 2: "z"}
        dir_axis = dir_map.get(dir_index)

        return normal, dir_axis

    def _calculate_plane_features(self, coors: list) -> tuple:
        """Calculate the normal and the direction of the plane"""
        if len(coors) < 3:
            raise ValueError("At least 3 coordinates are required.")

        # Calculate the normal using the shoelace formula
        normal, dir_axis = self._calculate_plane_normal(coors)

        # Sort the coordinates
        node_ids = MeshGeom.sort_anticlockwise(
            {j: coors[j] for j in range(len(coors))}, dir_axis
        )
        coors_sorted = [coors[j] for j in node_ids]

        # Calculate the perimeter and area
        perimeter = MeshGeom.calculate_perimeter(coors_sorted)
        area = MeshGeom.calculate_area(coors_sorted)
        return normal, perimeter, area

    def _calcuate_tetrahedron_volume(self, coors: list) -> float:
        """Calculate the volume of the tetrahedron"""
        if len(coors) != 4:
            raise ValueError("At least 4 coordinates are required.")

        matrix = np.array(
            [coors[1] - coors[0], coors[2] - coors[0], coors[3] - coors[0]]
        )
        volume = abs(np.linalg.det(matrix)) / 6.0
        return volume

    def _calculate_hexahedron_volume(self, coors: list, center: Coordinate) -> float:
        """Calculate the volume of the hexahedron"""
        if len(coors) != 8:
            raise ValueError("At least 8 coordinates are required.")

        east_west = []
        for coor in coors.values():
            if abs(coor.x - center.x) < 1e-10:
                east_west.append(coor)

        north_south = []
        for coor in coors.values():
            if abs(coor.y - center.y) < 1e-10:
                north_south.append(coor)

        top_bottom = []
        for coor in coors.values():
            if abs(coor.z - center.z) < 1e-10:
                top_bottom.append(coor)

        edge1 = MeshGeom.calculate_distance(east_west[0], east_west[1])
        edge2 = MeshGeom.calculate_distance(north_south[0], north_south[1])
        edge3 = MeshGeom.calculate_distance(top_bottom[0], top_bottom[1])
        volume = edge1 * edge2 * edge3
        return volume

    @property
    def dimension(self) -> str:
        """Return the dimension of the mesh, either 2d or 3d."""
        return self._dimension

    def refine_cells(self, indexes: list):
        pass

    def relax_cells(self, indexes: list):
        pass


class AdaptiveRectangularMesh(GenericMesh):
    """Adaptive rectangular mesh.

    When changing the mesh, the mesh will be refined firstly and then relaxed.
    """

    def __init__(self, nodes: list, faces: list, cells: list):
        """Adaptive rectangular mesh.

        Args:
            nodes: The list of nodes cooridnates.
            faces: The list of faces with their nodes indexes.
            cells: The list of cells with their faces indexes.
        """
        super().__init__(nodes, faces, cells)
        self._check_mesh_type()

        self._max_node_id = self.node_indexes[-1]
        self._max_face_id = self.face_indexes[-1]
        self._max_cell_id = self.cell_indexes[-1]

        self._sub_faces = {}
        for face in self.faces:
            self._sub_faces[face.id] = {"childrens": [], "parent": face.id}

        self._sub_cells = {}
        for cell in self.cells:
            self._sub_cells[cell.id] = {"childrens": [], "parent": cell.id}

        self._splited_faces = {}
        self._splited_cells = {}

        self._cell_levels = {cid: 0 for cid in self.cell_indexes}
        self._max_level = 0

    def _check_mesh_type(self):
        """Check the mesh type."""
        if self.dimension == "2d" and any(
            [len(cell.faces) != 4 for cell in self.cells]
        ):
            raise ValueError("Requires a rectangular mesh.")
        if self.dimension == "3d" and any(
            [len(cell.faces) != 8 for cell in self.cells]
        ):
            raise ValueError("Requires a rectangular mesh.")

    @property
    def sub_cells(self) -> dict:
        """Return the sub-cells of the mesh."""
        return self._sub_cells

    def refine_cells(self, indexes: list):
        for level in range(self._max_level, -1, -1):
            for cid in indexes:
                if self._cell_levels[cid] < level:
                    continue
                self._refine_cell(cid)

                neighbours = self._topo.collect_cell_neighbours(cid)
                for nb in neighbours:
                    if abs(self._cell_levels[nb] - level - 1) <= 1:
                        continue
                    if nb in indexes and self._cell_levels[nb] == level - 1:
                        continue
                    self._refine_cell(nb)

        self.refresh_mesh()

    def _refine_cell(self, index: int):
        """Refine the given cell."""
        if self.dimension == "2d":
            self._refine_2d_cell(index)
        else:
            self._refine_3d_cell(index)

    def _refine_2d_cell(self, index: int):
        """Refine the given 2D cell."""
        cell = self.cells[index]
        self._splited_cells[index] = cell

        faces = [self.faces[fid] for fid in cell.faces]
        self._splited_faces.update({fid: face for fid, face in zip(cell.faces, faces)})

        faces_new, nodes_new = [], []
        for face in faces:
            # Split the face line.
            n1, n2 = face.nodes
            med = (self.nodes[n1].coordinate + self.nodes[n2].coordinate) / 2
            node = Node(self._max_node_id + 1, med)
            nodes_new.append(node)
            self._nodes.append(node)
            self._max_node_id += 1

            # Split new faces.
            f1 = Face(
                self._max_face_id + 1,
                med,
                [n1, node.id],
                face.perimeter / 2,
                face.perimeter / 2,
                face.normal,
            )
            faces_new.append(f1)
            self._faces.append(f1)

            f2 = Face(
                self._max_face_id + 2,
                med,
                [node.id, n2],
                face.perimeter / 2,
                face.perimeter / 2,
                face.normal,
            )
            faces_new.append(f2)
            self._faces.append(f2)

            # Record raw face.
            self._sub_faces[face.id]["childrens"] = [f1.id, f2.id]
            self._sub_faces[f1.id] = {"childrens": [], "parent": face.id}
            self._sub_faces[f2.id] = {"childrens": [], "parent": face.id}

            self._splited_faces[face.id] = face
            self._faces.remove(face)
            self._max_face_id += 2

            # Update adjacent cell topology.
            cells = self._topo.collect_face_cells(face.id)
            for cid in cells:
                if cid == index:
                    continue
                adj_cell = self.cells[cid]
                adj_cell.faces.remove(face.id)
                adj_cell.faces.append(f1.id)
                adj_cell.faces.append(f2.id)

        # Create center node.
        nodes = self._topo.collect_cell_nodes(index)
        center = self._geom.calculate_center(nodes)
        node = Node(self._max_node_id + 1, center)
        nodes_new.append(node)
        self._nodes.append(node)
        self._max_node_id += 1

        face_centers = []
        for node in nodes_new:
            f_center = (node.coordinate + center.coordinate) / 2
            face_centers.append(f_center)

        # Create new faces
        for i in range(1, 5):
            f_id = self._max_face_id + i
            f_center = face_centers[i - 1]
            f_nodes = [nodes_new[i - 1].id, nodes_new[-1].id]

            parral_face = faces_new[(i + 1) % 8]
            face = Face(
                f_id,
                f_center,
                f_nodes,
                parral_face.perimeter,
                parral_face.area,
                parral_face.normal,
            )
            self._faces.append(face)
            self._sub_faces[f_id] = {"childrens": [], "parent": parral_face.id}
        self._max_face_id += 4

        # Create new cells.
        surface = cell.surface / 4
        volume = cell.volume / 4
        cell_faces = [
            [faces_new[0].id, faces_new[8].id, faces_new[11].id, faces_new[7].id],
            [faces_new[1].id, faces_new[2].id, faces_new[9].id, faces_new[8].id],
            [faces_new[9].id, faces_new[3].id, faces_new[4].id, faces_new[10].id],
            [faces_new[10].id, faces_new[5].id, faces_new[6].id, faces_new[11].id],
        ]
        for i in range(1, 5):
            c_id = self._max_cell_id + i
            c = Cell(
                c_id,
                center,
                cell_faces[i - 1],
                surface,
                volume,
            )
            self._cells.append(c)
            self._sub_cells[index]["childrens"].append(c_id)
            self._sub_cells[c_id] = {"childrens": [], "parent": index}
            self._cell_levels[c_id] = self._cell_levels[index] + 1
            self._max_cell_id += 1

    def _refine_3d_cell(self, index: int):
        """Refine the given 3D cell."""
        raise NotImplementedError()

    def relax_cells(self, indexes: list):
        relaxed = []
        for level in range(self._max_level, -1, -1):
            for cid in indexes:
                if self._cell_levels[cid] < level:
                    continue
                parent = self._sub_cells[cid]["parent"]
                if parent in relaxed:
                    continue
                relaxed.append(parent)
                self._relax_cell(cid)

                neighbours = self._topo.collect_cell_neighbours(cid)
                for nb in neighbours:
                    if abs(self._cell_levels[nb] - level + 1) <= 1:
                        continue
                    if nb in indexes and self._cell_levels[nb] == level + 1:
                        continue

                    sub_parent = self._sub_cells[nb]["parent"]
                    if sub_parent in relaxed:
                        continue
                    relaxed.append(sub_parent)
                    self._relax_cell(nb)
        self.refresh_mesh()

    def _relax_cell(self, index: int):
        """Relax the given cell."""
        if self.dimension == "2d":
            self._relax_2d_cell(index)
        else:
            self._relax_3d_cell(index)

    def _relax_2d_cell(self, index: int):
        """Relax the given 2D cell."""
        # Parent cell.
        parent = self._sub_cells[index]["parent"]
        parent_cell = self._splited_cells[parent]
        parent_faces = list(
            set([self._splited_faces[fid] for fid in parent_cell.faces])
        )
        parent_nodes = list(set([face.nodes for face in parent_faces]))

        # Sub cells.
        subs = self._sub_cells[parent]["childrens"]
        sub_faces, sub_nodes = [], []
        for sub in subs:
            sub_cell = self.cells[sub]
            sub_faces.extend([self.faces[fid] for fid in sub_cell.faces])
            sub_nodes.extend([face.nodes for face in sub_faces])

        sub_faces = list(set(sub_faces))
        sub_nodes = list(set(sub_nodes))

        # Update the adjacent cells.
        for cid in subs:
            adj_cells = self._topo.collect_cell_neighbours(cid)
            for adj in adj_cells:
                if adj in subs:
                    continue

                adj_cell = self.cells[adj]
                raw_faces = []
                for fid in adj_cell.faces:
                    face = self.faces[fid]
                    if face in sub_faces:
                        parent_face = self._sub_faces[fid]["parent"]
                        raw_faces.append(parent_face)
                    else:
                        raw_faces.append(fid)
                adj_cell.faces = list(set(raw_faces))

        # Remove the sub nodes.
        for node in sub_nodes:
            if node not in parent_nodes:
                self._nodes.remove(self.nodes[node])

        # Recover the parent faces.
        for face in sub_faces:
            self._faces.remove(face)
            self._splited_faces.pop(face.id)
            self._sub_faces.pop(face.id)

        for face in parent_faces:
            self._faces.append(face)
            self._splited_faces.pop(face.id)

        # Recover the parent cell.
        for cid in subs:
            self._cells.remove(self.cells[cid])
            self._splited_cells.pop(cid)
            self._sub_cells.pop(cid)

        self._cells.append(parent_cell)
        self._splited_cells.pop(parent)

    def _relax_3d_cell(self, index: int):
        """Relax the given 3D cell."""
        raise NotImplementedError()

    def refresh_mesh(self):
        """Refresh the mesh."""
        self._version += 1
        logger.info(f"Refreshing the mesh to {self._version}.")

        self._topos = None
        self._topo = self.get_topo_assistant()

        self._geom = None
        self._geom = self.get_geom_assistant()

        self._groups = {}


if __name__ == "__main__":
    from core.numerics.mesh import Grid2D

    low_left, upper_right = Coordinate(0, 0), Coordinate(100, 100)
    nx, ny = 5, 5
    grid = Grid2D(low_left, upper_right, nx, ny)

    nodes = [node.coordinate.to_np() for node in grid.nodes]
    faces = [face.nodes for face in grid.faces]
    cells = [cell.faces for cell in grid.cells]
