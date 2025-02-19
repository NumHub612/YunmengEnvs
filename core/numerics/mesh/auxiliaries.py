# -*- encoding: utf-8 -*-
"""
Copyright (C) 2025, The YunmengEnvs Contributors. Join us, share your ideas!  

Auxiliary functions for mesh processing.
"""
from core.numerics.mesh import Coordinate
from core.numerics.fields import Vector
import numpy as np
import math


class MeshTopo:
    """Mesh topology class for describing the topology.

    NOTE:
        - This class is lazily, i.e. it will not be initialized until it is needed.
    """

    def __init__(self, mesh: "Mesh"):
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

    def get_mesh(self) -> "Mesh":
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
                self._boundary_nodes.extend(self._mesh.get_face(fid).nodes)
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
                        nid1 = self._mesh.get_face(faces[i]).nodes[0]
                        for j in range(i + 1, len(faces)):
                            nid2 = self._mesh.get_face(faces[j]).nodes[0]
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
                    for nid in self._mesh.get_face(fid).nodes:
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
                    for nid in self._mesh.get_face(fid).nodes:
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

    def __init__(self, mesh: "Mesh"):
        self._mesh = mesh

        self._cell_to_cell_dists = None
        self._cell_to_face_dists = None
        self._cell_to_node_dists = None
        self._node_to_node_dists = None

        self._cell_to_cell_vects = None
        self._cell_to_face_vects = None

        self._stats = {"node": {}, "face": {}, "cell": {}}

    def get_mesh(self) -> "Mesh":
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
                        self._mesh.get_cell(cid).coordinate,
                        self._mesh.get_cell(j).coordinate,
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
                        cell.coordinate, self._mesh.get_face(fid).coordinate
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
                        self._mesh.get_cell(cid).coordinate,
                        self._mesh.get_node(j).coordinate,
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
                        self._mesh.get_node(nid).coordinate,
                        self._mesh.get_node(j).coordinate,
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
                        self._mesh.get_cell(cid).coordinate
                        - self._mesh.get_cell(j).coordinate
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
