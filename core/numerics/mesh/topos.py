# -*- encoding: utf-8 -*-
"""
Copyright (C) 2025, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Auxiliary functions for mesh processing.
"""
from core.numerics.mesh.elements import MeshDim
from configs.settings import logger

import collections
from scipy.spatial import cKDTree
import numpy as np


class MeshTopo:
    """Mesh topology class for describing the topology.

    Note:
        - All properties express the topological relationships within
        the mesh through element ids.
        - Support uncontinuous indecies for AMR meshes.
    """

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
    # region boundary properties
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
            ids, cells = zip(
                *((f.id, len(self.face_cells[f.id])) for f in self._mesh.faces)
            )
            size = len(ids)
            ids = np.fromiter(ids, dtype=np.intp, count=size)
            cells = np.fromiter(cells, dtype=np.intp, count=size)
            self._boundary_faces = ids[cells == 1].tolist()
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
            self._boundary_cells = [
                self.face_cells[fid][0] for fid in self.boundary_faces
            ]
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
    # region indices properties
    # -----------------------------------------------

    @property
    def face_indices(self) -> dict[int, int]:
        """Return the indices of faces with their ids."""
        if self._face_indices is None:
            self._face_indices = {f.id: i for i, f in enumerate(self._mesh.faces)}
        return self._face_indices

    @property
    def node_indices(self) -> dict[int, int]:
        """Return the indices of nodes with their ids."""
        if self._node_indices is None:
            self._node_indices = {n.id: i for i, n in enumerate(self._mesh.nodes)}
        return self._node_indices

    @property
    def cell_indices(self) -> dict[int, int]:
        """Return the indices of cells with their ids."""
        if self._cell_indices is None:
            self._cell_indices = {c.id: i for i, c in enumerate(self._mesh.cells)}
        return self._cell_indices

    # -----------------------------------------------
    # region connect properties
    # -----------------------------------------------

    @property
    def node_neighbours(self) -> dict[int, list[int]]:
        """Retrun the neighbours id of each node."""
        if self._mesh.dimension == MeshDim.DIM1:
            logger.warning("1D meshes do not check neighbours.")
            return None

        if self._node_neighbours is None:
            # Expand all edges by circular adjacency at once
            edges = (
                (u, v)
                for f in self._mesh.faces
                for u, v in self._face_to_edges(f.nodes)
            )
            edges = np.array(list(edges), dtype=np.intp)

            # Bidirectional edges + NumPy grouping
            both = np.vstack((edges, edges[:, ::-1]))
            order = both[:, 0].argsort()
            both_sorted = both[order]
            unq, idx = np.unique(both_sorted[:, 0], return_inverse=True)

            # Group node's neighbours
            size1, size2 = len(unq), len(both_sorted)
            self._node_neighbours = {}
            for i, nid in enumerate(unq):
                start = idx[i]
                stop = idx[i + 1] if i < size1 - 1 else size2
                nbrs = both_sorted[start:stop, 1].tolist()
                self._node_neighbours[nid] = nbrs

        return self._node_neighbours

    def _face_to_edges(self, nodes: list) -> list:
        """Return the edges from face nodes."""
        n = len(nodes)
        return [(nodes[i], nodes[(i + 1) % n]) for i in range(n)]

    @property
    def face_cells(self) -> dict[int, list[int]]:
        """Return the cells id connected to each face.

        Sorted to (left, right) or (owner, neighbour).
        """
        if self._face_cells is None:
            face_cells = collections.defaultdict(list)
            for c in self._mesh.cells:
                for f in c.faces:
                    face_cells[f].append(c.id)
            face_cells = {fid: list(set(cids)) for fid, cids in face_cells.items()}

            for fid, cids in face_cells.items():
                if len(cids) == 1:
                    continue
                face_cells[fid] = self._sort_face_cells(fid, cids)
            self._face_cells = face_cells
        return self._face_cells

    def _sort_face_cells(self, fid: int, cids: list) -> list:
        """Sort the cells id by dot product with face normal."""
        face = self._mesh.faces[fid]
        nodes = [self._mesh.nodes[n].coordinate for n in face.nodes]
        c0 = self._mesh.cells[cids[0]].coordinate
        c1 = self._mesh.cells[cids[1]].coordinate

        if len(cids) == 2:
            v0 = (nodes[1] - nodes[0]).to_np()
            v1 = (c0 - nodes[0]).to_np()
            res = v0[0] * v1[1] - v0[1] * v1[0]
            if res < 0:
                cids = [cids[1], cids[0]]
        else:
            v0 = nodes[1] - nodes[0]
            v1 = nodes[2] - nodes[0]
            normal = np.cross(v0.to_np(), v1.to_np())

            face_center = face.coordinate
            v0 = (c0 - face_center).to_np()
            v1 = (c1 - face_center).to_np()

            dot0 = np.dot(normal, v0)
            dot1 = np.dot(normal, v1)
            if dot0 > dot1:
                cids = [cids[1], cids[0]]
        return cids

    @property
    def node_faces(self) -> dict[int, list[int]]:
        """Return the faces id connected to each node."""
        if self._node_faces is None:
            node_faces = collections.defaultdict(list)
            for f in self._mesh.faces:
                for n in f.nodes:
                    node_faces[n].append(f.id)
            node_faces = {nid: list(set(fids)) for nid, fids in node_faces.items()}
            self._node_faces = node_faces
        return self._node_faces

    @property
    def node_cells(self) -> dict[int, list[int]]:
        """Return the cells id connected to each node."""
        if self._node_cells is None:
            face_nodes = {f.id: f.nodes for f in self._mesh.faces}
            tmp = collections.defaultdict(list)
            for c in self._mesh.cells:
                for f in c.faces:
                    for n in face_nodes[f]:
                        tmp[n].append(c.id)
            self._node_cells = {
                nid: list(dict.fromkeys(cids)) for nid, cids in tmp.items()
            }
        return self._node_cells

    @property
    def cell_nodes(self) -> dict[int, list[int]]:
        """Return the nodes id connected to each cell."""
        if self._cell_nodes is None:
            face_nodes = {f.id: f.nodes for f in self._mesh.faces}
            tmp = collections.defaultdict(list)
            for cell in self._mesh.cells:
                for fid in cell.faces:
                    for nid in face_nodes[fid]:
                        tmp[cell.id].append(nid)
            self._cell_nodes = {
                nid: list(dict.fromkeys(nids)) for nid, nids in tmp.items()
            }
        return self._cell_nodes

    @property
    def cell_neighbours(self) -> dict[int, list[int]]:
        """Return the neighbours of each cell."""
        if self._cell_neighbours is None:
            tmp = collections.defaultdict(list)
            for cids in self.face_cells.values():
                cids = list(cids)
                if len(cids) == 2:
                    tmp[cids[0]].append(cids[1])
                    tmp[cids[1]].append(cids[0])
            self._cell_neighbours = tmp
        return self._cell_neighbours
