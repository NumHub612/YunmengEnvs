# -*- encoding: utf-8 -*-
"""
Copyright (C) 2024, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Abstract mesh class for describing the geometry and topology.
"""
from core.numerics.mesh.elements import Coordinate, Element, Node, Face, Cell
from core.numerics.mesh.auxiliaries import MeshTopo, MeshGeom
from core.numerics.types import MeshDim, ElementType

from abc import ABC, abstractmethod
import numpy as np
import torch
import os
import pickle
from shapely.geometry import Polygon


class Mesh(ABC):
    """Abstract mesh class for describing the topology.

    - The element IDs in the mesh are required to be consecutively numbered,
    except for AMR types.
    - MESH currently only has three levels of objects: node, face, and cell.
    """

    def __init__(self):
        self._version = 1
        self._dim = MeshDim.NONE
        self._orthogonal = False

        self._nodes = []
        self._faces = []
        self._cells = []
        self._groups = {}

        self._topo = MeshTopo(self)
        self._geom = MeshGeom(self)

    def save(self, file_path: str):
        """Save the mesh instance."""
        with open(file_path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(file_path: str) -> "Mesh":
        """Load the mesh instance."""
        with open(file_path, "rb") as f:
            mesh = pickle.load(f)
        return mesh

    # -----------------------------------------------
    # region properties
    # -----------------------------------------------

    @property
    def version(self) -> int:
        """Return the mesh version."""
        return self._version

    @property
    def dimension(self) -> MeshDim:
        """Return mesh dimension."""
        return self._dim

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
    # region mesh query methods
    # -----------------------------------------------

    def get_nodes(self, nodes_ids: list[int]) -> list[Node]:
        """Get the nodes with the given ids."""
        return [self._nodes[i] for i in nodes_ids]

    def get_faces(self, faces_ids: list[int]) -> list[Face]:
        """Get the faces with the given ids."""
        return [self._faces[i] for i in faces_ids]

    def get_cells(self, cells_ids: list[int]) -> list[Cell]:
        """Get the cells with the given ids."""
        return [self._cells[i] for i in cells_ids]

    # -----------------------------------------------
    # region modification methods
    # -----------------------------------------------

    @abstractmethod
    def update(self, mask_indices: list[int]):
        """Update the mesh with the given mask indices.

        The mask is an array with the same length as the number of cells.
        Each element corresponds to a cell:

        + 1 indicates the cell should be refined,
        + 0 indicates the cell should remain unchanged.
        + -1 indicates the cell should be coarsened,
        """
        pass

    # -----------------------------------------------
    # region additional methods
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
            raise ValueError("Invalid group ids.")
        self._groups[group_id] = (etype, ids)

    def delete_group(self, group_id: str):
        """Delete the given name group."""
        if group_id in self._groups:
            self._groups.pop(group_id)

    def get_group(self, group_id: str) -> tuple[ElementType, list]:
        """Return the element ids of given group."""
        if group_id not in self._groups:
            return None
        return self._groups[group_id]

    def get_all_groups(self) -> dict[str, tuple]:
        """Return all groups."""
        return self._groups

    # -----------------------------------------------
    # region extension methods
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


# -----------------------------------------------
# region --- GenericMesh ---
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
            center = MeshGeom.calculate_center(nodes)

            if len(node_ids) == 2:
                self._dimension = MeshDim.DIM2
            else:
                self._dimension = MeshDim.DIM3
                normal = self._calculate_plane_normal(nodes)
                nodes = MeshTopo.sort_anticlockwise(nodes)
                node_ids = [n.id for n in nodes]
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
            center = MeshGeom.calculate_center(faces)

            if self._dimension == MeshDim.DIM2:
                normal = self._calculate_plane_normal(faces)
                faces = MeshTopo.sort_anticlockwise(faces)
                face_ids = [f.id for f in faces]
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
# region --- GridMesh ---
# -----------------------------------------------


class GridMesh(Mesh):
    """Catesian cut-cell and boundary refined grid mesh."""

    def __init__(self, rect: tuple, poly: Polygon, max_depth: int = 6):
        pass
