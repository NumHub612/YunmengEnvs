# -*- encoding: utf-8 -*-
"""
Copyright (C) 2025, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Auxiliary functions for mesh processing.
"""
from core.numerics.mesh.elements import Coordinate, Element, Cell
from core.numerics.fields import Vector, Field
from core.numerics.types import MeshDim
import numpy as np
import math


class MeshTopo:
    """Mesh topology class for describing the topology."""

    def __init__(self, mesh):
        self.reset(mesh)

    def get_mesh(self):
        """Return the bounded mesh."""
        return self._mesh

    def reset(self, mesh):
        """Reset the assistant."""
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

        self._face_indices = None
        self._node_indices = None
        self._cell_indices = None

    # -----------------------------------------------
    # --- boundaray and interior properties ---
    # -----------------------------------------------

    @property
    def boundary_nodes(self) -> list[int]:
        """Return the ids of boundary nodes."""
        if self._boundary_nodes is None:
            bound_faces = self._mesh.get_faces(self.boundary_faces)
            bound_nodes = [f.nodes for f in bound_faces]
            bound_nodes = np.unique(np.concatenate(bound_nodes))
            self._boundary_nodes = bound_nodes.tolist()
        return self._boundary_nodes

    @property
    def interior_nodes(self) -> list[int]:
        """Return the ids of interior nodes."""
        if self._interior_nodes is None:
            nodes_id = [node.id for node in self._mesh.nodes]
            self._interior_nodes = list(
                set(nodes_id) - set(self.boundary_nodes),
            )
        return self._interior_nodes

    @property
    def boundary_faces(self) -> list[int]:
        """Return the ids of boundary faces."""
        if self._boundary_faces is None:
            bound_faces = []
            for face in self._mesh.faces:
                if len(self.face_cells[face.id]) == 1:
                    bound_faces.append(face.id)
            self._boundary_faces = bound_faces
        return self._boundary_faces

    @property
    def interior_faces(self) -> list[int]:
        """Return the ids of interior faces."""
        if self._interior_faces is None:
            faces_id = [face.id for face in self._mesh.faces]
            self._interior_faces = list(
                set(faces_id) - set(self.boundary_faces),
            )
        return self._interior_faces

    @property
    def boundary_cells(self) -> list[int]:
        """Return the ids of boundary cells."""
        if self._boundary_cells is None:
            self._boundary_cells = []
            for fid in self.boundary_faces:
                cids = self.face_cells(fid)
                self._boundary_cells.append(cids[0])
        return self._boundary_cells

    @property
    def interior_cells(self) -> list[int]:
        """Return the ids of interior cells."""
        if self._interior_cells is None:
            cells_id = [cell.id for cell in self._mesh.cells]
            self._interior_cells = list(
                set(cells_id) - set(self.boundary_cells),
            )
        return self._interior_cells

    # -----------------------------------------------
    # --- connectivity properties ---
    # -----------------------------------------------

    @property
    def node_neighbours(self) -> dict:
        """Retrun the neighbours id of each node."""
        if self._node_neighbours is None:
            neighbours = {n.id: set() for n in self._mesh.nodes}
            if self._mesh.dimension == MeshDim.DIM1:
                print("Warning: 1D mesh neighbour searching.")
                return None
            for face in self._mesh.faces:
                nodes = face.nodes  # sorted list of nodes
                for i in range(len(nodes)):
                    for j in range(i + 1, len(nodes)):
                        neighbours[nodes[i]].add(nodes[j])
                        neighbours[nodes[j]].add(nodes[i])
            self._node_neighbours = neighbours
        return self._node_neighbours

    @property
    def face_cells(self) -> dict:
        """Return the cells id connected to each face."""
        if self._face_cells is None:
            faces_id = [face.id for face in self._mesh.faces]
            face_cells = {fid: set() for fid in faces_id}
            for cell in self._mesh.cells:
                for fid in cell.faces:
                    face_cells[fid].add(cell.id)
            self._face_cells = face_cells
        return self._face_cells

    @property
    def node_faces(self) -> dict:
        """Return the faces id connected to each node."""
        if self._node_faces is None:
            node_faces = {n.id: set() for n in self._mesh.nodes}
            for face in self._mesh.faces:
                for nid in face.nodes:
                    node_faces[nid].add(face.id)
            self._node_faces = node_faces
        return self._node_faces

    @property
    def node_cells(self) -> dict:
        """Return the cells id connected to each node."""
        if self._node_cells is None:
            face_nodes = {f.id: f.nodes for f in self._mesh.faces}
            node_cells = {n.id: set() for n in self._mesh.nodes}
            for cell in self._mesh.cells:
                for fid in cell.faces:
                    for nid in face_nodes[fid]:
                        node_cells[nid].add(cell.id)
            self._node_cells = node_cells
        return self._node_cells

    @property
    def cell_nodes(self) -> dict:
        """Return the nodes id connected to each cell."""
        if self._cell_nodes is None:
            face_nodes = {f.id: f.nodes for f in self._mesh.faces}
            cell_nodes = {c.id: set() for c in self._mesh.cells}
            for cell in self._mesh.cells:
                for fid in cell.faces:
                    for nid in face_nodes[fid]:
                        cell_nodes[cell.id].add(nid)
            self._cell_nodes = cell_nodes
        return self._cell_nodes

    @property
    def cell_neighbours(self) -> dict:
        """Return the neighbours of each cell."""
        if self._cell_neighbours is None:
            cell_neighbours = {c.id: set() for c in self._mesh.cells}
            for cells in self.face_cells.values():
                cells = list(cells)
                if len(cells) == 2:
                    cell_neighbours[cells[0]].add(cells[1])
                    cell_neighbours[cells[1]].add(cells[0])
            self._cell_neighbours = cell_neighbours
        return self._cell_neighbours

    # -----------------------------------------------
    # --- indices methods ---
    # -----------------------------------------------

    @property
    def face_indices(self) -> dict:
        """Return the indices of faces with their ids."""
        if self._face_indices is None:
            self._face_indices = {f.id: i for i, f in enumerate(self._mesh.faces)}
        return self._face_indices

    @property
    def node_indices(self) -> dict:
        """Return the indices of nodes with their ids."""
        if self._node_indices is None:
            self._node_indices = {n.id: i for i, n in enumerate(self._mesh.nodes)}
        return self._node_indices

    @property
    def cell_indices(self) -> dict:
        """Return the indices of cells with their ids."""
        if self._cell_indices is None:
            self._cell_indices = {c.id: i for i, c in enumerate(self._mesh.cells)}
        return self._cell_indices

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
            dist = node.coordinate.to_np() - coordinate.to_np()
            dist = np.linalg.norm(dist)
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
        min_cell_id = -1
        for cell in self._mesh.cells:
            dist = cell.coordinate.to_np() - coordinate.to_np()
            dist = np.linalg.norm(dist)
            if dist <= max_dist and dist < min_dist:
                min_dist = dist
                min_cell_id = cell.id
        return min_cell_id

    def search_nearest_face(
        self,
        coordinate: Coordinate,
        max_dist: float,
    ) -> int:
        """Search the nearest face to the given coordinate."""
        min_dist = float("inf")
        min_face_id = -1
        for face in self._mesh.faces:
            dist = face.coordinate.to_np() - coordinate.to_np()
            dist = np.linalg.norm(dist)
            if dist <= max_dist and dist < min_dist:
                min_dist = dist
                min_face_id = face.id
        return min_face_id

    # -----------------------------------------------
    # --- generation methods ---
    # -----------------------------------------------

    def generate_projection(
        self,
        coordinate: Coordinate,
        face_index: int,
    ) -> Coordinate:
        """Generate the projection on the given face."""
        face = self._mesh.faces[face_index]
        normal = face.normal
        vec_np = (coordinate - face.coordinate).to_np()
        proj_np = np.dot(vec_np, normal.to_np()) * normal.to_np()
        proj_np = proj_np + face.coordinate.to_np()
        proj_coord = Coordinate.from_np(proj_np)
        return proj_coord


class MeshGeom:
    """Mesh geometry class for calculating the geometry."""

    def __init__(self, mesh):
        self.reset(mesh)

    def get_mesh(self):
        """Return the bounded mesh."""
        return self._mesh

    def reset(self, mesh):
        """Reset the assistant."""
        self._mesh = mesh
        self._topo = mesh.get_topo_assistant()

        self._cell2cell_dists = None
        self._cell2face_dists = None
        self._cell2node_dists = None
        self._node2node_dists = None

        self._face_areas = None
        self._face_perimeters = None
        self._face_normals = None

        self._cell_volumes = None
        self._cell_surfaces = None

        self._cell2cell_vects = None
        self._cell2face_vects = None

    # -----------------------------------------------
    # --- static geometry methods ---
    # -----------------------------------------------

    @staticmethod
    def calculate_distance(
        point1: Coordinate | Element, point2: Coordinate | Element
    ) -> float:
        """Calculate the distance between two coordinates."""
        if isinstance(point1, Element):
            point1 = point1.coordinate
        if isinstance(point2, Element):
            point2 = point2.coordinate
        return np.linalg.norm(point1.to_np() - point2.to_np())

    @staticmethod
    def calculate_center(points: list) -> Coordinate:
        """Calculate the center of the given coordinates."""
        coords = [
            point.coordinate if isinstance(point, Element) else point
            for point in points
        ]

        return Coordinate.from_np(
            np.mean([coord.to_np() for coord in coords], axis=0),
        )

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
    def calculate_area(points: list) -> float:
        """Calculate the area of the given coordinates.

        Note:
            The coordinates should be sorted.
        """
        if len(points) < 3:
            raise ValueError("At least 3 points are required.")

        coords = [
            point.coordinate if isinstance(point, Element) else point
            for point in points
        ]

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
    def extract_coordinates_separated(
        elements: list[Element],
        dims: str = "xyz",
    ) -> dict:
        """Extract the coordinates of each element separatedly."""
        dims = dims.lower()
        if dims not in ["xyz", "xy", "xz", "yz", "x", "y", "z"]:
            raise ValueError(f"Invalid dimension: {dims}.")

        xs = np.array([e.coordinate.x for e in elements])
        ys = np.array([e.coordinate.y for e in elements])
        zs = np.array([e.coordinate.z for e in elements])

        coordinate_map = {"x": xs, "y": ys, "z": zs}
        coordinates = {d: coordinate_map.get(d) for d in dims}
        return coordinates

    # -----------------------------------------------
    # --- face properties ---
    # For the List properties, the order of the results is the same
    # as the order of the faces in mesh.
    # -----------------------------------------------

    @property
    def face_areas(self) -> list:
        """Return the areas of each face."""
        if self._face_areas is None:
            if self._mesh.dimension == MeshDim.DIM1:
                face_areas = [0.0] * self._mesh.face_count
            elif self._mesh.dimension == MeshDim.DIM2:
                # Use the perimeter as the area for 2D mesh
                face_areas = self.face_perimeters
            else:
                face_areas = self._calculate_areas()
            self._face_areas = face_areas
        return self._face_areas

    def _calculate_areas(self):
        face_areas = [0.0] * self._mesh.face_count
        for i, face in enumerate(self._mesh.faces):
            nodes = self._mesh.get_nodes(face.nodes)
            area = self.calculate_area(nodes)
            face_areas[i] = area
        return face_areas

    @property
    def face_perimeters(self) -> list:
        """Return the perimeters of each face."""
        if self._face_perimeters is None:
            if self._mesh.dimension == MeshDim.DIM1:
                face_perimeters = [0.0] * self._mesh.face_count
            elif self._mesh.dimension == MeshDim.DIM2:
                face_perimeters = self._calculate_perimeters_2d()
            else:
                face_perimeters = self._calculate_perimeters_3d()
            self._face_perimeters = face_perimeters
        return self._face_perimeters

    def _calculate_perimeters_2d(self):
        face_perimeters = [0.0] * self._mesh.face_count
        for i, face in enumerate(self._mesh.faces):
            nodes = self._mesh.get_nodes(face.nodes)
            dist = self.calculate_distance(nodes[0], nodes[1])
            face_perimeters[i] = dist
        return face_perimeters

    def _calculate_perimeters_3d(self):
        face_perimeters = [0.0] * self._mesh.face_count
        for i, face in enumerate(self._mesh.faces):
            nodes = self._mesh.get_nodes(face.nodes)
            perimeter = sum(
                self.calculate_distance(nodes[j], nodes[(j + 1) % len(nodes)])
                for j in range(len(nodes))
            )
            face_perimeters[i] = perimeter
        return face_perimeters

    @property
    def face_normals(self) -> list:
        """Return the normals of each face."""
        if self._face_normals is None:
            if self._mesh.dimension == MeshDim.DIM1:
                face_normals = [None] * self._mesh.face_count
            elif self._mesh.dimension == MeshDim.DIM2:
                face_normals = self._calculate_normals_2d()
            else:
                face_normals = self._calculate_normals_3d()
            self._face_normals = face_normals
        return self._face_normals

    def _calculate_normals_2d(self):
        face_normals = [None] * self._mesh.face_count
        for i, face in enumerate(self._mesh.faces):
            nodes = self._mesh.get_nodes(face.nodes)
            normal = np.cross(
                (nodes[1].coordinate - nodes[0].coordinate).to_np(),
                [0, 0, 1],
            )
            normal /= np.linalg.norm(normal)
            face_normals[i] = normal
        return face_normals

    def _calculate_normals_3d(self):
        face_normals = [None] * self._mesh.face_count
        for i, face in enumerate(self._mesh.faces):
            nodes = self._mesh.get_nodes(face.nodes)
            normal = np.cross(
                (nodes[1].coordinate - nodes[0].coordinate).to_np(),
                (nodes[2].coordinate - nodes[1].coordinate).to_np(),
            )
            normal /= np.linalg.norm(normal)
            face_normals[i] = normal
        return face_normals

    # -----------------------------------------------
    # --- cell properties ---
    # For the List properties, the order of the results is the same
    # as the order of the cells in mesh.
    # -----------------------------------------------

    @property
    def cell_volumes(self) -> list:
        """Return the volumes of each cell."""
        if self._cell_volumes is None:
            if self._mesh.dimension == MeshDim.DIM1:
                cell_volumes = [0.0] * self._mesh.cell_count
            elif self._mesh.dimension == MeshDim.DIM2:
                cell_volumes = self._calculate_volumes_2d()
            else:
                cell_volumes = self._calculate_volumes_3d()
            self._cell_volumes = cell_volumes
        return self._cell_volumes

    def _calculate_volumes_2d(self):
        cell_areas = [0.0] * self._mesh.cell_count
        for i, cell in enumerate(self._mesh.cells):
            nodes = self._topo.cell_nodes[cell.id]
            nodes = self._mesh.get_nodes(nodes)
            area = self.calculate_area(nodes)
            cell_areas[i] = area
        return cell_areas

    def _calculate_volumes_3d(self):
        cell_volumes = [0.0] * self._mesh.cell_count

        cell = self._mesh.cells[0]
        if len(cell.faces) == 4:
            func = self._calcuate_tetrahedron_volume
        elif len(cell.faces) == 6:
            func = self._calculate_hexahedron_volume
        else:
            raise ValueError("Unsupported cell shape.")

        for i, cell in enumerate(self._mesh.cells):
            volume = func(cell)
            cell_volumes[i] = volume
        return cell_volumes

    def _calcuate_tetrahedron_volume(self, cell: Cell):
        nodes = self._topo.cell_nodes[cell.id]
        nodes = self._mesh.get_nodes(nodes)
        coors = [node.coordinate for node in nodes]
        if len(cell.faces) != 4 or len(nodes) != 4:
            raise ValueError("Invalid tetrahedron.")

        matrix = np.array(
            [
                coors[1] - coors[0],
                coors[2] - coors[0],
                coors[3] - coors[0],
            ]
        )
        volume = abs(np.linalg.det(matrix)) / 6.0
        return volume

    def _calculate_hexahedron_volume(self, cell: Cell):
        faces = self._topo.cell_faces[cell.id]
        nodes = self._topo.cell_nodes[cell.id]
        if len(faces) != 6 or len(nodes) != 8:
            raise ValueError("Invalid hexahedron.")

        edges = set()
        for i in range(3):
            nodes = self._mesh.get_nodes(faces[i].nodes)
            side1 = self.calculate_distance(nodes[0], nodes[1])
            side2 = self.calculate_distance(nodes[1], nodes[2])
            edges.add(side1)
            edges.add(side2)

        volume = edges.pop() * edges.pop() * edges.pop()
        return volume

    @property
    def cell_surfaces(self) -> list:
        """Return the surfaces of each cell."""
        if self._cell_surfaces is None:
            if self._mesh.dimension == MeshDim.DIM1:
                cell_surfaces = [0.0] * self._mesh.cell_count
            else:
                cell_surfaces = self._calculate_cell_surface()
            self._cell_surfaces = cell_surfaces
        return self._cell_surfaces

    def _calculate_cell_surface(self):
        cell_surfaces = [0.0] * self._mesh.cell_count
        id_indices = self._topo.face_indices
        for i, cell in enumerate(self._mesh.cells):
            surface = sum(self.face_areas[id_indices[f.id]] for f in cell.faces)
            cell_surfaces[i] = surface
        return cell_surfaces

    # -----------------------------------------------
    # --- distance properties ---
    # -----------------------------------------------

    @property
    def cell2cell_distances(self) -> dict:
        """Return the distances between each pair of cells."""
        if self._cell2cell_dists is None:
            cell_dists = {c.id: {} for c in self._mesh.cells}
            for cell in self._mesh.cells:
                neighbours = self._topo.cell_neighbours[cell.id]
                neighbours = self._mesh.get_cells(neighbours)
                for nb in neighbours:
                    dist = self.calculate_distance(cell, nb)
                    cell_dists[cell.id][nb.id] = dist
                    cell_dists[nb.id][cell.id] = dist
            self._cell2cell_dists = cell_dists
        return self._cell2cell_dists

    @property
    def cell2face_distances(self) -> dict:
        """Return the distances between each cell and its faces."""
        if self._cell2face_dists is None:
            cell_face_dists = {c.id: {} for c in self._mesh.cells}
            for cell in self._mesh.cells:
                faces = self._mesh.get_faces(cell.faces)
                for face in faces:
                    dist = self.calculate_distance(cell, face)
                    cell_face_dists[cell.id][face.id] = dist
            self._cell2face_dists = cell_face_dists
        return self._cell2face_dists

    @property
    def cell2node_distances(self) -> dict:
        """Return the distances between each cell and its nodes."""
        if self._cell2node_dists is None:
            cell_node_dists = {c.id: {} for c in self._mesh.cells}
            for cell in self._mesh.cells:
                nodes = self._topo.cell_nodes[cell.id]
                nodes = self._mesh.get_nodes(nodes)
                for node in nodes:
                    dist = self.calculate_distance(cell, node)
                    cell_node_dists[cell.id][node.id] = dist
            self._cell2node_dists = cell_node_dists
        return self._cell2node_dists

    @property
    def node2node_distances(self) -> dict:
        """Return the distances between each pair of nodes."""
        if self._node2node_dists is None:
            node_dists = {n.id: {} for n in self._mesh.nodes}
            for node in self._mesh.nodes:
                neighbours = self._topo.node_neighbours[node.id]
                neighbours = self._mesh.get_nodes(neighbours)
                for nb in neighbours:
                    dist = self.calculate_distance(node, nb)
                    node_dists[node.id][nb.id] = dist
                    node_dists[nb.id][node.id] = dist
            self._node2node_dists = node_dists
        return self._node2node_dists

    # -----------------------------------------------
    # --- vector properties ---
    # -----------------------------------------------

    @property
    def cell2cell_vectors(self) -> dict:
        """Return the unit vectors from each cell to its neighbours."""
        if self._cell2cell_vects is None:
            cell_vecs = {c.id: {} for c in self._mesh.cells}
            for cell in self._mesh.cells:
                neighbours = self._topo.cell_neighbours[cell.id]
                neighbours = self._mesh.get_cells(neighbours)
                for nb in neighbours:
                    vec_np = (cell.coordinate - nb.coordinate).to_np()
                    vec = Vector.from_np(vec_np)
                    vec /= vec.magnitude
                    cell_vecs[cell.id][nb.id] = vec
                    cell_vecs[nb.id][cell.id] = vec  # Symmetric matrix
            self._cell2cell_vects = cell_vecs
        return self._cell2cell_vects

    @property
    def cell2face_vectors(self) -> dict:
        """Return the unit vectors from each cell to its faces."""
        if self._cell2face_vects is None:
            cell_face_vecs = {c.id: {} for c in self._mesh.cells}
            for cell in self._mesh.cells:
                faces = self._mesh.get_faces(cell.faces)
                for face in faces:
                    vec_np = (cell.coordinate - face.coordinate).to_np()
                    vec = Vector.from_np(vec_np)
                    vec /= vec.magnitude
                    cell_face_vecs[cell.id][face.id] = vec
            self._cell2face_vects = cell_face_vecs
        return self._cell2face_vects
