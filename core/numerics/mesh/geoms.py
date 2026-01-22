# -*- encoding: utf-8 -*-
"""
Copyright (C) 2025, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Auxiliary functions for mesh processing.
"""
from core.numerics.mesh.elements import Cell, Face, Node, MeshDim
from core.numerics.mesh.tools import calculate_distance, calculate_area
from core.numerics.fields import Vector
import numpy as np
import enum


class GeomType(enum.Enum):
    """The geometry types."""

    IdBased = 0
    Point = 1
    Polyline = 2
    Polygon = 3
    Polyhedron = 4


class MeshGeom:
    """Mesh geometry class for calculating the geometry.

    Note:
        - For `face_areas`, `face_perimeters`, `face_normals`,
        they are all storaged as a list corresponding to the order of `mesh.faces`.
        - For `face_areas`, in 2D mesh, the area is the perimeter.
        - For `face_normals`, in 2D mesh, normal is left cell to right one.
        - For `cell_volumes`, `cell_surfaces`, they are all storaged as a list
        corresponding to the order of `mesh.cells`.
        - For `cell_volumes`, in 2D mesh, the volume is the area.
        - For all `_distances`, `_vectors`, they are storaged as a two-layers
        dictionary, with both keys being element ids.
    """

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
    # region face properties
    # For the List properties, the order of the results is the same
    # as the order of the faces in mesh.
    # -----------------------------------------------

    @property
    def face_area(self) -> list[float]:
        """Return the area of each face."""
        if self._face_areas is None:
            if self._mesh.dimension == MeshDim.DIM1:
                face_areas = [0.0] * self._mesh.face_count
            elif self._mesh.dimension == MeshDim.DIM2:
                # Use the perimeter as the area for 2D mesh
                face_areas = self.face_perimeter
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
    def face_perimeter(self) -> list:
        """Return the perimeter of each face."""
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
            dist = calculate_distance(nodes[0], nodes[1])
            face_perimeters[i] = dist
        return face_perimeters

    def _calculate_perimeters_3d(self):
        face_perimeters = [0.0] * self._mesh.face_count
        for i, face in enumerate(self._mesh.faces):
            nodes = self._mesh.get_nodes(face.nodes)
            perimeter = sum(
                calculate_distance(nodes[j], nodes[(j + 1) % len(nodes)])
                for j in range(len(nodes))
            )
            face_perimeters[i] = perimeter
        return face_perimeters

    @property
    def face_normal(self) -> list[Vector]:
        """Return the normal of each face."""
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
            # Calculate the normal using the cross product of two vectors
            # The normal is always pointing outward(left to right)
            normal = np.cross(
                (nodes[1].coordinate - nodes[0].coordinate).to_np(),
                [0, 0, 1],
            )
            normal /= np.linalg.norm(normal)
            face_normals[i] = Vector.from_data(normal)
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
            face_normals[i] = Vector.from_data(normal)
        return face_normals

    # -----------------------------------------------
    # region cell properties
    # For the List properties, the order of the results is the same
    # as the order of the cells in mesh.
    # -----------------------------------------------

    @property
    def cell_volume(self) -> list:
        """Return the volume of each cell."""
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
            area = calculate_area(nodes)
            cell_areas[i] = area
        return cell_areas

    def _calculate_volumes_3d(self):
        cell_volumes = [0.0] * self._mesh.cell_count

        cell = self._mesh.cells[0]
        if len(cell.faces) == 4:
            func = self._tetrahedron_volume
        elif len(cell.faces) == 6:
            func = self._hexahedron_volume
        else:
            raise ValueError("Unsupported cell shape.")

        for i, cell in enumerate(self._mesh.cells):
            volume = func(cell)
            cell_volumes[i] = volume
        return cell_volumes

    def _tetrahedron_volume(self, cell: Cell):
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

    def _hexahedron_volume(self, cell: Cell):
        faces = self._topo.cell_faces[cell.id]
        nodes = self._topo.cell_nodes[cell.id]
        if len(faces) != 6 or len(nodes) != 8:
            raise ValueError("Invalid hexahedron.")

        edges = set()
        for i in range(3):
            nodes = self._mesh.get_nodes(faces[i].nodes)
            side1 = calculate_distance(nodes[0], nodes[1])
            side2 = calculate_distance(nodes[1], nodes[2])
            edges.add(side1)
            edges.add(side2)

        volume = edges.pop() * edges.pop() * edges.pop()
        return volume

    @property
    def cell_surface(self) -> list:
        """Return the surface of each cell."""
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
            surface = sum(self.face_area[id_indices[f]] for f in cell.faces)
            cell_surfaces[i] = surface
        return cell_surfaces

    # -----------------------------------------------
    # region distance properties
    # -----------------------------------------------

    @property
    def cell2cell_distance(self) -> dict:
        """Return the distances between each pair of cells."""
        if self._cell2cell_dists is None:
            cell_dists = {c.id: {} for c in self._mesh.cells}
            for cell in self._mesh.cells:
                neighbours = self._topo.cell_neighbours[cell.id]
                neighbours = self._mesh.get_cells(neighbours)
                for nb in neighbours:
                    dist = calculate_distance(cell, nb)
                    cell_dists[cell.id][nb.id] = dist
                    cell_dists[nb.id][cell.id] = dist
            self._cell2cell_dists = cell_dists
        return self._cell2cell_dists

    @property
    def cell2face_distance(self) -> dict:
        """Return the distances between each cell and its faces."""
        if self._cell2face_dists is None:
            cell_face_dists = {c.id: {} for c in self._mesh.cells}
            for cell in self._mesh.cells:
                faces = self._mesh.get_faces(cell.faces)
                for face in faces:
                    dist = calculate_distance(cell, face)
                    cell_face_dists[cell.id][face.id] = dist
            self._cell2face_dists = cell_face_dists
        return self._cell2face_dists

    @property
    def cell2node_distance(self) -> dict:
        """Return the distances between each cell and its nodes."""
        if self._cell2node_dists is None:
            cell_node_dists = {c.id: {} for c in self._mesh.cells}
            for cell in self._mesh.cells:
                nodes = self._topo.cell_nodes[cell.id]
                nodes = self._mesh.get_nodes(nodes)
                for node in nodes:
                    dist = calculate_distance(cell, node)
                    cell_node_dists[cell.id][node.id] = dist
            self._cell2node_dists = cell_node_dists
        return self._cell2node_dists

    @property
    def node2node_distance(self) -> dict:
        """Return the distances between each pair of nodes."""
        if self._node2node_dists is None:
            node_dists = {n.id: {} for n in self._mesh.nodes}
            for node in self._mesh.nodes:
                neighbours = self._topo.node_neighbours[node.id]
                neighbours = self._mesh.get_nodes(neighbours)
                for nb in neighbours:
                    dist = calculate_distance(node, nb)
                    node_dists[node.id][nb.id] = dist
                    node_dists[nb.id][node.id] = dist
            self._node2node_dists = node_dists
        return self._node2node_dists

    # -----------------------------------------------
    # region vector properties
    # -----------------------------------------------

    @property
    def cell2cell_vector(self) -> dict:
        """Return the unit vectors from each cell to its neighbours."""
        if self._cell2cell_vects is None:
            cell_vecs = {c.id: {} for c in self._mesh.cells}
            for cell in self._mesh.cells:
                neighbours = self._topo.cell_neighbours[cell.id]
                neighbours = self._mesh.get_cells(neighbours)
                for nb in neighbours:
                    vec_np = (cell.coordinate - nb.coordinate).to_np()
                    vec = Vector.from_data(vec_np)
                    vec /= vec.magnitude
                    cell_vecs[cell.id][nb.id] = vec
            self._cell2cell_vects = cell_vecs
        return self._cell2cell_vects

    @property
    def cell2face_vector(self) -> dict[str, dict[str, Vector]]:
        """Return the unit vectors from each cell to its faces."""
        if self._cell2face_vects is None:
            cell_face_vecs = {c.id: {} for c in self._mesh.cells}
            for cell in self._mesh.cells:
                faces = self._mesh.get_faces(cell.faces)
                for face in faces:
                    vec_np = (cell.coordinate - face.coordinate).to_np()
                    vec = Vector.from_data(vec_np)
                    vec /= vec.magnitude
                    cell_face_vecs[cell.id][face.id] = vec
            self._cell2face_vects = cell_face_vecs
        return self._cell2face_vects
