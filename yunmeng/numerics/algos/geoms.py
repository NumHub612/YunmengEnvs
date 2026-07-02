# -*- encoding: utf-8 -*-
"""
Copyright (C) 2025, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Auxiliary functions for mesh processing.
"""

from yunmeng.numerics.enums import MeshDimension
from yunmeng.numerics.mesh import Face, Mesh, Element, Coordinate
from yunmeng.numerics.fields.variables import Variable, Var
from yunmeng.numerics.algos.topos import (
    MeshTopo,
    sort_anticlockwise,
    calculate_center,
    extract_coordinates,
)

import numpy as np
from typing import Dict, List, Optional

# -----------------------------------------------
# region geom methods
# -----------------------------------------------


def calculate_distance(
    point1: Coordinate | Element, point2: Coordinate | Element
) -> float:
    """Calculate the distance between two coordinates."""
    if isinstance(point1, Element):
        point1 = point1.coordinate
    if isinstance(point2, Element):
        point2 = point2.coordinate
    return np.linalg.norm(point1.to_numpy() - point2.to_numpy())


def generate_projection(
    coordinate: Coordinate,
    face: Face,
    normal: Variable,
) -> Coordinate:
    """Generate the projection on the given face."""
    vec_np = (coordinate - face.coordinate).to_numpy()
    proj_np = np.dot(vec_np, normal.to_numpy()) * normal.to_numpy()
    proj_np = proj_np + face.coordinate.to_numpy()
    proj_coord = Coordinate.from_numpy(proj_np)
    return proj_coord


# -----------------------------------------------
# region MeshGeom
# -----------------------------------------------


class MeshGeom:
    """Mesh geometry assistant.

    NOTE:
    - All properties are stored corresponding to the mesh's topology.
    """

    def __init__(self, mesh: Mesh):
        self._mesh: Mesh = mesh
        self._topo: MeshTopo = mesh.get_topo_assistant()

        # Element properties caches
        self._face_areas: Optional[np.ndarray] = None
        self._face_perimeters: Optional[np.ndarray] = None
        self._face_normals: Optional[np.ndarray] = None
        self._cell_volumes: Optional[np.ndarray] = None
        self._cell_surfaces: Optional[np.ndarray] = None

        # Distance caches
        self._cell2cell_dists: List[Dict[int, float]] = None
        self._cell2face_dists: List[Dict[int, float]] = None
        self._cell2node_dists: List[Dict[int, float]] = None

        # statistics(min, max, mean)
        self._node_dists_stats: tuple[float, float, float] = None
        self._cell_dists_stats: tuple[float, float, float] = None

        # Vector caches
        self._cell2cell_vects: List[Dict[int, Variable]] = None
        self._cell2face_vects: List[Dict[int, Variable]] = None

    def reset(self, mesh: Mesh):
        """Reset the assistant."""
        self.__init__(mesh)

    # -----------------------------------------------
    # region Continous properties
    # -----------------------------------------------

    @property
    def face_area(self) -> np.ndarray:
        """Face area."""
        if self._face_areas is None:
            if self._mesh.dimension != MeshDimension.D3:
                face_areas = self.face_perimeter
            else:
                face_areas = self._calculate_areas_3d()
            self._face_areas = np.array(face_areas)
        return self._face_areas

    def _calculate_areas_3d(self) -> List[float]:
        face_areas = []
        for face in self._mesh.faces:
            nodes = self._mesh.get_nodes(face.nodes)
            coords = extract_coordinates(nodes)
            if len(coords) == 3:  # Triangle
                v1 = coords[1] - coords[0]
                v2 = coords[2] - coords[0]
                area = 0.5 * np.linalg.norm(np.cross(v1, v2))
            elif len(coords) == 4:  # Quad
                d1 = coords[2] - coords[0]
                d2 = coords[3] - coords[1]
                area = 0.5 * np.linalg.norm(np.cross(d1, d2))
            else:  # General polygon
                area, v0 = 0.0, coords[0]
                for i in range(1, len(coords) - 1):
                    v1 = coords[i]
                    v2 = coords[i + 1]
                    area += 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0))
            face_areas.append(area)
        return face_areas

    @property
    def face_perimeter(self) -> np.ndarray:
        """Face perimeter."""
        if self._face_perimeters is None:
            if self._mesh.dimension == MeshDimension.NONE:
                face_perimeters = [0.0] * self._mesh.face_count
            elif self._mesh.dimension != MeshDimension.D3:
                face_perimeters = self._calculate_perimeters_2d()
            else:
                face_perimeters = self._calculate_perimeters_3d()
            self._face_perimeters = np.array(face_perimeters)
        return self._face_perimeters

    def _calculate_perimeters_2d(self) -> List[float]:
        face_perimeters = []
        for face in self._mesh.faces:
            # In 2D, face connects two nodes
            n1_idx, n2_idx = face.nodes[0], face.nodes[1]
            n1 = self._mesh.nodes[n1_idx]
            n2 = self._mesh.nodes[n2_idx]
            dist = calculate_distance(n1, n2)
            face_perimeters.append(dist)
        return face_perimeters

    def _calculate_perimeters_3d(self) -> List[float]:
        face_perimeters = []
        for fid in range(self._mesh.face_count):
            node_ids = self._topo.face_nodes[fid]
            nodes = self._mesh.get_nodes(node_ids)
            coords = extract_coordinates(nodes)
            perimeter = sum(
                np.linalg.norm(coords[j] - coords[(j + 1) % len(coords)])
                for j in range(len(coords))
            )
            face_perimeters.append(perimeter)
        return face_perimeters

    @property
    def face_normal(self) -> np.ndarray:
        """Face unit normal vectors."""
        if self._face_normals is None:
            if self._mesh.dimension == MeshDimension.NONE:
                face_normals = [None] * self._mesh.face_count
            elif self._mesh.dimension != MeshDimension.D3:
                face_normals = self._calculate_normals_2d()
            else:
                face_normals = self._calculate_normals_3d()
            self._face_normals = np.array(face_normals)
        return self._face_normals

    def _calculate_normals_2d(self) -> List[Variable]:
        face_normals = []
        for fid in range(self._mesh.face_count):
            n1_id, n2_id = self._topo.face_nodes[fid]
            n1 = self._mesh.nodes[n1_id]
            n2 = self._mesh.nodes[n2_id]
            edge = (n2.coordinate - n1.coordinate).to_numpy()
            normal = np.array([edge[1], -edge[0], 0.0])
            magnitude = np.linalg.norm(normal)
            if magnitude > 1e-12:
                normalized_normal = normal / magnitude
            else:  # Degenerate face
                normalized_normal = np.array([0.0, 1.0, 0.0])
            face_normals.append(Var(normalized_normal))
        return face_normals

    def _calculate_normals_3d(self) -> List[Variable]:
        face_normals = []
        for fid in range(self._mesh.face_count):
            node_ids = self._topo.face_nodes[fid]
            nodes = self._mesh.get_nodes(node_ids)
            coords = extract_coordinates(nodes)
            if len(coords) < 3:  # Degenerate
                face_normals.append(Var([0.0, 0.0, 0.0]))
                continue
            v1 = coords[1] - coords[0]
            v2 = coords[2] - coords[0]
            normal = np.cross(v1, v2)
            magnitude = np.linalg.norm(normal)
            if magnitude > 1e-12:
                normalized_normal = normal / magnitude
            else:  # Degenerate face
                normalized_normal = np.array([0.0, 0.0, 1.0])
            face_normals.append(Var(normalized_normal))
        return face_normals

    @property
    def cell_volume(self) -> np.ndarray:
        """Cell volumes."""
        if self._cell_volumes is None:
            if self._mesh.dimension == MeshDimension.D1:
                cell_volumes = [0.0] * self._mesh.cell_count
            elif self._mesh.dimension == MeshDimension.D2:
                cell_volumes = self._calculate_volumes_2d()
            else:
                cell_volumes = self._calculate_volumes_3d()
            self._cell_volumes = np.array(cell_volumes)
        return self._cell_volumes

    def _calculate_volumes_2d(self) -> List[float]:
        """Calculate the cell area as volume."""
        cell_volumes = []
        for cid, _ in enumerate(self._mesh.cells):
            node_ids = self._topo.cell_nodes[cid]
            nodes = self._mesh.get_nodes(node_ids)
            nodes, _ = sort_anticlockwise(nodes)
            coords = extract_coordinates(nodes)

            x = coords[:, 0]
            y = coords[:, 1]

            # Add the first point to the end to form a closed polygon
            x_appended = np.append(x, x[0])
            y_appended = np.append(y, y[0])

            # Shoelly's formula
            area = 0.5 * np.abs(
                np.sum(x_appended[:-1] * y_appended[1:])
                - np.sum(x_appended[1:] * y_appended[:-1])
            )
            cell_volumes.append(area)
        return cell_volumes

    def _calculate_volumes_3d(self) -> List[float]:
        # Placeholder for 3D volume calculation
        cell_volumes = []
        for cid, cell in enumerate(self._mesh.cells):
            face_ids = cell.faces
            if len(face_ids) == 4:  # Tetrahedron
                node_ids = self._topo.cell_nodes[cid]
                if len(node_ids) != 4:
                    raise ValueError(
                        f"Cell {cid} has 4 faces but {len(node_ids)} nodes, invalid Tet."
                    )
                nodes = self._mesh.get_nodes(node_ids)
                coords = extract_coordinates(nodes)
                matrix = np.array(
                    [
                        coords[1] - coords[0],
                        coords[2] - coords[0],
                        coords[3] - coords[0],
                    ]
                )
                volume = abs(np.linalg.det(matrix)) / 6.0
            elif len(face_ids) == 6:  # Hexahedron
                node_ids = self._topo.cell_nodes[cid]
                nodes = self._mesh.get_nodes(node_ids)
                coords = extract_coordinates(nodes)
                mins = np.min(coords, axis=0)
                maxs = np.max(coords, axis=0)
                volume = np.prod(maxs - mins)  # Approximate volume
            else:
                raise ValueError(
                    f"Unsupported 3D cell type with {len(face_ids)} faces."
                )
            cell_volumes.append(volume)
        return cell_volumes

    @property
    def cell_surface(self) -> np.ndarray:
        """Cell surface areas."""
        if self._cell_surfaces is None:
            if self._mesh.dimension == MeshDimension.D1:
                cell_surfaces = [0.0] * self._mesh.cell_count
            else:
                cell_surfaces = self._calculate_cell_surface()
            self._cell_surfaces = np.array(cell_surfaces)
        return self._cell_surfaces

    def _calculate_cell_surface(self) -> List[float]:
        cell_surfaces = [0.0] * self._mesh.cell_count
        face_areas = self.face_area
        for cid, cell in enumerate(self._mesh.cells):
            surface = sum(face_areas[f] for f in cell.faces)
            cell_surfaces[cid] = surface
        return cell_surfaces

    # -----------------------------------------------
    # region non-Continous attrs
    # -----------------------------------------------

    @property
    def cell2cell_distance(self) -> List[Dict[int, float]]:
        """Distances between each cell and its neighbours."""
        if self._cell2cell_dists is None:
            cell_dists_list = [dict() for _ in range(self._mesh.cell_count)]
            coords = extract_coordinates(self._mesh.cells)
            for cid in range(self._mesh.cell_count):
                nbr_ids = self._topo.cell_neighbours[cid]
                own_coord = coords[cid]
                nbr_coords = coords[nbr_ids]

                # Vectorized distance calculation
                diffs = own_coord - nbr_coords
                dists = np.linalg.norm(diffs, axis=1)

                # Store by neighbor ID
                dists_map = {nbr: dist for nbr, dist in zip(nbr_ids, dists)}
                cell_dists_list[cid] = dists_map

            self._cell2cell_dists = cell_dists_list
        return self._cell2cell_dists

    @property
    def cell2face_distance(self) -> List[Dict[int, float]]:
        """Distances between each cell and its face centers."""
        if self._cell2face_dists is None:
            num_cells = self._mesh.cell_count
            cell_coords = extract_coordinates(self._mesh.cells)
            face_coords = extract_coordinates(self._mesh.faces)

            cell_face_dists_list = [dict() for _ in range(num_cells)]
            for cid in range(num_cells):
                face_ids = self._topo.cell_faces[cid]
                own_coord = cell_coords[cid]
                face_coords_subset = face_coords[face_ids]

                # Vectorized distance calculation
                diffs = own_coord - face_coords_subset
                dists = np.linalg.norm(diffs, axis=1)

                # Store by face ID
                dists_map = {fid: dist for fid, dist in zip(face_ids, dists)}
                cell_face_dists_list[cid] = dists_map

            self._cell2face_dists = cell_face_dists_list
        return self._cell2face_dists

    @property
    def cell2cell_vector(self) -> List[Dict[int, Variable]]:
        """Unit vectors from each cell to its neighbours."""
        if self._cell2cell_vects is None:
            num_cells = self._mesh.cell_count
            coords = extract_coordinates(self._mesh.cells)

            cell_vectors_list = [dict() for _ in range(num_cells)]
            for cid in range(num_cells):
                nbr_ids = self._topo.cell_neighbours[cid]
                own_coord = coords[cid]
                nbr_coords = coords[nbr_ids]

                # Vectorized vector calculation: from nbr to own
                vectors_raw = own_coord - nbr_coords
                # Correct direction: from own to nbr
                vectors_raw = -vectors_raw

                # Calculate magnitudes
                magnitudes = np.linalg.norm(vectors_raw, axis=1, keepdims=True)
                magnitudes = np.where(magnitudes == 0, 1.0, magnitudes)
                # Normalize
                unit_vectors = vectors_raw / magnitudes

                # Convert to Variable and store by neighbor ID
                vectors_map = {
                    nbr: Var(unit_vectors[i]) for i, nbr in enumerate(nbr_ids)
                }
                cell_vectors_list[cid] = vectors_map

            self._cell2cell_vects = cell_vectors_list
        return self._cell2cell_vects

    @property
    def cell2face_vector(self) -> List[Dict[int, Variable]]:
        """Unit vectors from each cell to its face centers."""
        if self._cell2face_vects is None:
            num_cells = self._mesh.cell_count
            cell_coords = extract_coordinates(self._mesh.cells)
            face_coords = extract_coordinates(self._mesh.faces)

            cell_face_vectors_list = [dict() for _ in range(num_cells)]
            for cid in range(num_cells):
                face_ids = self._topo.cell_faces[cid]
                own_coord = cell_coords[cid]
                face_coords_subset = face_coords[face_ids]

                # Vectorized vector calculation: from own to face
                vectors_raw = face_coords_subset - own_coord

                # Calculate magnitudes for normalization
                magnitudes = np.linalg.norm(vectors_raw, axis=1, keepdims=True)
                magnitudes = np.where(magnitudes == 0, 1.0, magnitudes)
                # Normalize
                unit_vectors = vectors_raw / magnitudes

                # Convert to Variable and store by face ID
                vectors_map = {
                    fid: Var(unit_vectors[i]) for i, fid in enumerate(face_ids)
                }
                cell_face_vectors_list[cid] = vectors_map

            self._cell2face_vects = cell_face_vectors_list
        return self._cell2face_vects

    def get_cell2cell_distance(self, cell_id: int, nbr_id: int) -> float:
        """Get the distance between a cell and its neighbour."""
        try:
            return self.cell2cell_distance[cell_id][nbr_id]
        except KeyError:
            return self.cell2cell_distance[nbr_id][cell_id]

    # -----------------------------------------------
    # region non-Continous statis
    # -----------------------------------------------

    @property
    def cell2cell_distance_stats(self) -> tuple[float, float, float]:
        """Statistics of cell-to-cell distances."""
        if self._cell_dists_stats is None:
            all_dists = []
            for cell_dists in self.cell2cell_distance:
                all_dists.extend(cell_dists.values())
            self._cell_dists_stats = (
                float(np.min(all_dists)),
                float(np.max(all_dists)),
                float(np.mean(all_dists)),
            )
        return self._cell_dists_stats
