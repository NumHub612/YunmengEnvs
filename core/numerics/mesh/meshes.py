# -*- encoding: utf-8 -*-
"""
Copyright (C) 2024, The YunmengEnvs Contributors. Join us, share your ideas!  

Abstract mesh class for describing the geometry and topology.
"""
from core.numerics.mesh import Coordinate, Node, Face, Cell
from core.numerics.mesh import MeshTopo, MeshGeom
from core.numerics.fields import Vector
from configs.settings import logger

from abc import ABC, abstractmethod
import numpy as np


class Mesh(ABC):
    """Abstract mesh class for describing the topology.

    Notes:
        - Don't support isolated element.
        - Don't support non-Cartesian coordinates.
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

    @property
    @abstractmethod
    def is_orthogonal(self) -> bool:
        """Return True if the mesh is orthogonal."""
        pass

    # -----------------------------------------------
    # --- mesh query methods ---
    # -----------------------------------------------

    def get_node(self, index: int) -> Node:
        """Get the node with the given index."""
        for node in self._nodes:
            if node.id == index:
                return node
        return None

    def get_face(self, index: int) -> Face:
        """Get the face with the given index."""
        for face in self._faces:
            if face.id == index:
                return face
        return None

    def get_cell(self, index: int) -> Cell:
        """Get the cell with the given index."""
        for cell in self._cells:
            if cell.id == index:
                return cell
        return None

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
        return self._dimension

    @property
    def is_orthogonal(self) -> bool:
        return False

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

    @property
    def is_orthogonal(self) -> bool:
        return True

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

        for cid in indexes:
            cell = self.cells[cid]
            faces = cell.faces
            adj_cells = self._topo.collect_cell_neighbours(cid)
            for adj_id in adj_cells:
                adj_cell = self.cells[adj_id]
                for face in adj_cell.faces:
                    if face in faces:
                        adj_cell.faces.remove(face)
            for face in faces:
                if face in self._faces:
                    self._faces.remove(face)
            self._cells.remove(cell)
            self._cell_levels.pop(cid)

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
            self._max_face_id += 2

            # Update adjacent cell topology.
            cells = self._topo.collect_face_cells(face.id)
            for cid in cells:
                if cid == index:
                    continue
                adj_cell = self.cells[cid]
                adj_cell.faces.append(f1.id)
                adj_cell.faces.append(f2.id)

        # Create center node.
        nodes_id = self._topo.collect_cell_nodes(index)
        nodes = [self.nodes[nid].coordinate for nid in nodes_id]
        center = self._geom.calculate_center(nodes)
        center_node = Node(self._max_node_id + 1, center)
        nodes_new.append(center_node)
        self._nodes.append(center_node)
        self._max_node_id += 1

        face_centers = []
        for node in nodes_new:
            f_center = (node.coordinate + center) / 2
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
            faces_new.append(face)
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
