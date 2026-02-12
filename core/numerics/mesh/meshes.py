# -*- encoding: utf-8 -*-
"""
Copyright (C) 2024, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Abstract mesh class for describing the geometry and topology.
"""
from core.numerics.enums import MeshDimension
from core.numerics.mesh.elements import Coordinate, Node, Face, Cell
from core.numerics.mesh.spatials import Mesh
from core.numerics.algos.topos import sort_anticlockwise, calculate_center

import numpy as np
import torch
from shapely.geometry import Polygon


# -----------------------------------------------
# region GenericMesh
# -----------------------------------------------


class GenericMesh(Mesh):
    """Generic mesh with fixed topology and geometry."""

    def __init__(
        self,
        nodes: np.ndarray | torch.Tensor,
        faces: np.ndarray | torch.Tensor,
        cells: np.ndarray | torch.Tensor,
    ):
        """Generic mesh.

        Args:
            nodes: The nodes cooridnates.
            faces: The faces with nodes indices.
            cells: The cells with faces indices.

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
        self._orthogonal = False

        self._nodes = self._generate_nodes(nodes)
        self._faces = self._generate_faces(faces)
        self._cells = self._generate_cells(cells)

    def _generate_nodes(self, nodes: list):
        """Generate the nodes of the mesh."""
        results = [None] * len(nodes)
        for i, coor in enumerate(nodes):
            # id is index.
            results[i] = Node(i, Coordinate(*coor))
        return results

    def _generate_faces(self, faces: list):
        """Generate the faces of the mesh."""
        normals = []
        results = [None] * len(faces)
        for i, node_ids in enumerate(faces):
            nodes = self.get_nodes(node_ids)
            center = calculate_center(nodes)

            if len(node_ids) == 2:
                self._dimension = MeshDimension.D2
            else:
                self._dimension = MeshDimension.D3
                normal = self._calculate_plane_normal(nodes)
                nodes, node_ids = sort_anticlockwise(nodes, node_ids)
                normals.append(normal)

            results[i] = Face(i, center, node_ids)

        if normals and all(sum(n) == 1 for n in normals):
            self._orthogonal = True
        return results

    def _generate_cells(self, cells: list):
        """Generate the cells of the mesh."""
        normals = []
        results = [None] * len(cells)
        for i, face_ids in enumerate(cells):
            faces = self.get_faces(face_ids)
            center = calculate_center(faces)

            if self._dimension == MeshDimension.D2:
                normal = self._calculate_plane_normal(faces)
                faces, face_ids = sort_anticlockwise(faces, face_ids)
                normals.append(normal)
            else:
                # only surport tetrahedron and hexahedron.
                if len(faces) not in [4, 8]:
                    raise ValueError("Unsupported cell type.")

            results[i] = Cell(i, center, face_ids)

        if normals and all(sum(n) == 1 for n in normals):
            self._orthogonal = True
        return results

    def _calculate_plane_normal(self, coords: list) -> tuple:
        """Calculate the normal of the plane"""
        if len(coords) < 3:
            raise ValueError("At least 3 coordinates are required.")

        # Calculate the normal using the shoelace formula
        coords = [c.coordinate.to_np() for c in coords]
        normal = np.cross(coords[1] - coords[0], coords[2] - coords[0])
        return normal

    def update(self, mask_indices: list[int]):
        raise NotImplementedError("Generic mesh cannot be updated.")


# -----------------------------------------------
# region GridMesh
# -----------------------------------------------


class GridMesh(Mesh):
    """Catesian cut-cell and boundary refined grid mesh."""

    def __init__(self, rect: tuple, poly: Polygon, max_depth: int = 6):
        pass
