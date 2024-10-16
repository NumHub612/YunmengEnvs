# -*- encoding: utf-8 -*-
"""
Copyright (C) 2024, The YunmengEnvs Contributors. Join us, for you talents!  

1d/2d/2d structured grids.
"""
from core.numerics.mesh import Mesh, Coordinate, Node, Face, Cell
from core.numerics.fields import Vector

import numpy as np


class Grid1D(Mesh):
    """1D uniform structured grid.

    NOTE:
        - The 1d grid is a special case of mesh, somehow it is viered.
    """

    def __init__(self, start_coord: Coordinate, end_coord: Coordinate, num: int):
        self._version = 1

        self._nodes = []
        self._faces = []
        self._cells = []

        self._groups = {}

        self._generate(start_coord, end_coord, num)

    def _generate(self, start, end, num):
        # generate nodes
        dx = (end.x - start.x) / (num - 1)
        for i in range(num):
            x = start.x + i * dx
            node = Node(i, Coordinate(x))
            self._nodes.append(node)

        # generate mesh
        for i in range(num - 1):
            node1, node2 = self._nodes[i : i + 2]
            length = abs(node2.coord.x - node1.coord.x)
            center = 0.5 * (node1.coord + node2.coord)
            normal = Vector(0, 1)

            # face
            face1 = Face(i, [i, i], node1.coord, 1, 1, normal)
            self._faces.append(face1)

            face2 = Face(i + 1, [i + 1, i + 1], node2.coord, 1, 1, normal)
            self._faces.append(face2)

            # cell
            cell = Cell(i, [i, i + 1], center, length, length)
            self._cells.append(cell)

    @property
    def version(self) -> int:
        return self._version

    @property
    def domain(self) -> str:
        return "1d"

    @property
    def extents(self) -> list:
        min_x, max_x, _ = self.stat_node_elevation
        return [min_x, max_x], None, None

    @property
    def node_count(self) -> int:
        return len(self._nodes)

    @property
    def nodes(self) -> list:
        return self._nodes

    @property
    def stat_node_elevation(self) -> tuple:
        xs = [node.coord.x for node in self._nodes]
        min_x = min(xs)
        max_x = max(xs)
        avg_x = np.mean(xs)
        return min_x, max_x, avg_x

    @property
    def face_count(self) -> int:
        return len(self._faces)

    @property
    def faces(self) -> list:
        return self._faces

    @property
    def stat_face_area(self) -> tuple:
        areas = [face.area for face in self._faces]
        min_area = min(areas)
        max_area = max(areas)
        avg_area = np.mean(areas)
        return min_area, max_area, avg_area

    @property
    def cell_count(self) -> int:
        return len(self._cells)

    @property
    def cells(self) -> list:
        return self._cells

    @property
    def stat_cell_volume(self) -> tuple:
        volumes = [cell.volume for cell in self._cells]
        min_vol = min(volumes)
        max_vol = max(volumes)
        avg_vol = np.mean(volumes)
        return min_vol, max_vol, avg_vol

    def refine_cell(self, index: int):
        pass

    def relax_cell(self, index: int):
        pass

    def set_node_group(self, group_name: str, node_indices: list):
        if group_name in self._groups:
            raise ValueError(f"Group {group_name} already exists.")

        self._groups[group_name] = node_indices

    def set_face_group(self, group_name: str, face_indices: list):
        if group_name in self._groups:
            raise ValueError(f"Group {group_name} already exists.")

        self._groups[group_name] = face_indices

    def set_cell_group(self, group_name: str, cell_indices: list):
        if group_name in self._groups:
            raise ValueError(f"Group {group_name} already exists.")

        self._groups[group_name] = cell_indices

    def get_group(self, group_name: str):
        return self._groups.get(group_name, [])

    def delete_group(self, group_name: str):
        if group_name in self._groups:
            self._groups.pop(group_name)


class Grid2D(Mesh):
    """2D structured grid."""

    def __init__(self, start, end, num):
        pass


class Grid3D(Mesh):
    """3D structured grid."""

    def __init__(self, start, end, num):
        pass
