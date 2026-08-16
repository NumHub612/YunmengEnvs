# -*- encoding: utf-8 -*-
"""
Copyright (C) 2024, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Abstract mesh class for describing the geometry and topology.
"""

from yunmeng.numerics.enums import MeshDimension
from yunmeng.numerics.mesh.elements import Coordinate, Node, Face, Cell
from yunmeng.numerics.mesh.mesh import Mesh
from yunmeng.numerics.mesh.helpers import sort_anticlockwise, calculate_center

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
            results[i] = Node(Coordinate(*coor))
        return np.array(results)

    def _generate_faces(self, faces: list):
        """Generate the faces of the mesh."""
        normals = []
        results = [None] * len(faces)
        for i, ids in enumerate(faces):
            nodes = self.get_nodes(ids)
            center = calculate_center(nodes)

            if len(ids) == 2:
                self._dim = MeshDimension.D2
            else:
                self._dim = MeshDimension.D3
                normal = self._calculate_plane_normal(nodes)
                nodes, ids = sort_anticlockwise(nodes, ids)
                normals.append(normal)

            results[i] = Face(center, ids)

        if normals and all(sum(n) == 1 for n in normals):
            self._orthogonal = True
        return np.array(results)

    def _generate_cells(self, cells: list):
        """Generate the cells of the mesh."""
        normals = []
        results = [None] * len(cells)
        for i, ids in enumerate(cells):
            faces = self.get_faces(ids)
            center = calculate_center(faces)

            if self._dim == MeshDimension.D2:
                normal = self._calculate_plane_normal(faces)
                faces, ids = sort_anticlockwise(faces, ids)
                normals.append(normal)
            else:
                # only surport tetrahedron and hexahedron.
                if len(faces) not in [4, 8]:
                    raise ValueError("Unsupported cell type.")

            results[i] = Cell(center, ids)

        if normals and all(sum(n) == 1 for n in normals):
            self._orthogonal = True
        return np.array(results)

    def _calculate_plane_normal(self, coords: list) -> tuple:
        """Calculate the normal of the plane"""
        if len(coords) < 3:
            raise ValueError("Require at least 3 coordinates.")

        # Calculate the normal using the shoelace formula
        cs = [c.coordinate.to_numpy() for c in coords]
        normal = np.cross(cs[1] - cs[0], cs[2] - cs[0])
        return normal


# -----------------------------------------------
# region GridMesh
# -----------------------------------------------
