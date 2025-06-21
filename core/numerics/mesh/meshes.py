# -*- encoding: utf-8 -*-
"""
Copyright (C) 2024, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Abstract mesh class for describing the geometry and topology.
"""
from core.numerics.mesh import Coordinate, Node, Face, Cell
from core.numerics.mesh import MeshTopo, MeshGeom
from core.numerics.types import MeshDim, ElementType

from abc import ABC, abstractmethod
import numpy as np
import torch
import os
import pickle


class Mesh(ABC):
    """Abstract mesh class for describing the topology."""

    def __init__(self):
        self._version = 1

        self._topo = None
        self._geom = None

    @abstractmethod
    def save(self, file_path: str):
        """Save the mesh to the given file path."""
        pass

    @staticmethod
    @abstractmethod
    def load(file_path: str) -> "Mesh":
        """Load the mesh from local."""
        pass

    # -----------------------------------------------
    # --- properties ---
    # -----------------------------------------------

    @property
    def version(self) -> int:
        """Return the mesh version."""
        return self._version

    @property
    @abstractmethod
    def dimension(self) -> MeshDim:
        """Return the mesh dimension."""
        pass

    @property
    @abstractmethod
    def orthogonal(self) -> bool:
        """Return if the mesh is orthogonal."""
        pass

    @property
    @abstractmethod
    def node_count(self) -> int:
        """Return the number of nodes."""
        pass

    @property
    @abstractmethod
    def nodes(self) -> list[Node]:
        """Return all nodes."""
        pass

    @nodes.setter
    @abstractmethod
    def nodes(self, nodes: list[Node]):
        """Reset the nodes."""
        pass

    @property
    @abstractmethod
    def face_count(self) -> int:
        """Return the number of faces."""
        pass

    @property
    @abstractmethod
    def faces(self) -> list[Face]:
        """Return all faces."""
        pass

    @property
    @abstractmethod
    def cell_count(self) -> int:
        """Return the number of cells."""
        pass

    @property
    @abstractmethod
    def cells(self) -> list[Cell]:
        """Return all cells."""
        pass

    # -----------------------------------------------
    # --- mesh query methods ---
    # -----------------------------------------------

    @abstractmethod
    def get_nodes(self, nodes_ids: list[int]) -> list[Node]:
        """Get the nodes with the given id."""
        pass

    @abstractmethod
    def get_faces(self, faces_ids: list[int]) -> list[Face]:
        """Get the faces with the given id."""
        pass

    @abstractmethod
    def get_cells(self, cells_ids: list[int]) -> list[Cell]:
        """Get the cells with the given id."""
        pass

    # -----------------------------------------------
    # --- mesh modification methods ---
    # -----------------------------------------------

    @abstractmethod
    def update(self, mask_indices: list[int]):
        """Update the mesh with the given mask indices.

        The mask is an array with the same length as the number of cells.
        Each element corresponds to a cell:

        + 1 indicates the cell should be refined,
        + -1 indicates the cell should be coarsened,
        + 0 indicates the cell should remain unchanged.
        """
        pass

    # -----------------------------------------------
    # --- additional methods ---
    # -----------------------------------------------

    @abstractmethod
    def set_group(
        self,
        etype: ElementType,
        group_name: str,
        indices: list,
    ):
        """Set the group with the given name and indices ."""
        pass

    @abstractmethod
    def delete_group(self, group_name: str):
        """Delete the given node group."""
        pass

    @abstractmethod
    def get_group(self, group_name: str) -> tuple[list, ElementType]:
        """Return the element indices of given group."""
        pass

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
        self._dim = MeshDim.NONE
        self._orthogonal = None

        self._nodes = self._generate_nodes(nodes)
        self._faces = self._generate_faces(faces)
        self._cells = self._generate_cells(cells)

        self._topo = MeshTopo(self)
        self._geom = MeshGeom(self)

        self._groups = {}

    def _generate_nodes(self, nodes: list):
        """Generate the nodes of the mesh."""
        nodes = []
        for i, coor in enumerate(nodes):
            # id is index.
            self._nodes.append(Node(i, Coordinate(*coor)))

    def _generate_faces(self, faces: list):
        """Generate the faces of the mesh."""
        normals = []
        for i, node_ids in enumerate(faces):
            coors = {j: self._nodes[j].coordinate for j in node_ids}
            center = MeshGeom.calculate_center(list(coors.values()))

            if len(node_ids) == 2:
                self._dimension = MeshDim.DIM2
            else:
                self._dimension = MeshDim.DIM3
                normal, dir_axis = self._calculate_plane_normal(list(coors.values()))
                node_ids = MeshGeom.sort_anticlockwise(coors, dir_axis)
                normals.append(normal)

            # id is index.
            self._faces.append(Face(i, center, node_ids))

        if normals and all(sum(n) == 1 for n in normals):
            self._orthogonal = True

    def _generate_cells(self, cells: list):
        """Generate the cells of the mesh."""
        normals = []
        for i, face_ids in enumerate(cells):
            faces = [self._faces[j] for j in face_ids]
            coors = {face.id: face.coordinate for face in faces}
            center = MeshGeom.calculate_center(list(coors.values()))

            if self._dimension == MeshDim.DIM2:
                normal, dir_axis = self._calculate_plane_normal(list(coors.values()))
                face_ids = MeshGeom.sort_anticlockwise(coors, dir_axis)
                normals.append(normal)
            else:
                # only surport tetrahedron and hexahedron.
                if len(faces) not in [4, 8]:
                    raise ValueError("Unsupported cell type.")

            # id is index.
            self._cells.append(Cell(i, center, face_ids))

        if normals and all(sum(n) == 1 for n in normals):
            self._orthogonal = True

    def _calculate_plane_normal(self, coors: list) -> tuple:
        """Calculate the normal of the plane"""
        if len(coors) < 3:
            raise ValueError("At least 3 coordinates are required.")

        # Calculate the normal using the shoelace formula
        normal = np.cross(
            (coors[1].to_np() - coors[0].to_np()),
            (coors[2].to_np() - coors[1].to_np()),
        )

        # Check the direction of the normal
        dir_index = np.argmax(np.abs(normal))
        dir_map = {0: "x", 1: "y", 2: "z"}
        dir_axis = dir_map.get(dir_index)

        return normal, dir_axis

    # -----------------------------------------------
    # --- properties ---
    # -----------------------------------------------

    @property
    def dimension(self) -> MeshDim:
        return self._dimension

    @property
    def orthogonal(self) -> bool:
        return False

    @property
    def node_count(self) -> int:
        return len(self._nodes)

    @property
    def nodes(self) -> list[Node]:
        return self._nodes

    @nodes.setter
    def nodes(self, nodes: list[Node]):
        for node in nodes:
            self._nodes[node.id] = node
        self._geom.reset()
        self._version += 1

    @property
    def face_count(self) -> int:
        return len(self._faces)

    @property
    def faces(self) -> list[Face]:
        return self._faces

    @property
    def cell_count(self) -> int:
        return len(self._cells)

    @property
    def cells(self) -> list[Cell]:
        return self._cells

    # -----------------------------------------------
    # --- methods ---
    # -----------------------------------------------

    def get_nodes(self, nodes_ids: list[int]) -> list[Node]:
        return [self._nodes[i] for i in nodes_ids]

    def get_faces(self, faces_ids: list[int]) -> list[Face]:
        return [self._faces[i] for i in faces_ids]

    def get_cells(self, cells_ids: list[int]) -> list[Cell]:
        return [self._cells[i] for i in cells_ids]

    def set_group(self, etype, group_name, indices):
        if not isinstance(etype, ElementType):
            raise ValueError("Invalid element type.")
        if group_name in self._groups:
            raise ValueError("Group already exists.")
        self._groups[group_name] = (indices, etype)

    def get_group(self, group_name):
        if group_name not in self._groups:
            raise ValueError("Group does not exist.")
        return self._groups[group_name]

    def delete_group(self, group_name):
        if group_name in self._groups:
            self._groups.pop(group_name)
        else:
            return None

    def save(self, file_path: str):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        mesh_file = os.path.join(file_path, "mesh.pkl")
        with open(mesh_file, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(file_path: str) -> Mesh:
        mesh_file = os.path.join(file_path, "mesh.pkl")
        with open(mesh_file, "rb") as f:
            mesh = pickle.load(f)
        return mesh
