# -*- encoding: utf-8 -*-
"""
Copyright (C) 2024, The YunmengEnvs Contributors. Join us, for you talents!  

1d/2d/2d structured grids.
"""
from core.numerics.mesh import Mesh, Coordinate, Node, Face, Cell
from core.numerics.fields import Vector
from configs.settings import logger

import numpy as np


class Grid1D(Mesh):
    """1D uniform structured grid.

    NOTE:
        - The 1d grid is a special case of mesh, somehow it is viered.
    """

    def __init__(self, start_coord: Coordinate, end_coord: Coordinate, num: int):
        """
        Initialize a 1D uniform structured grid.

        Args:
            start_coord: The starting coordinate of the grid.
            end_coord: The ending coordinate of the grid.
            num: The number of nodes in the grid.
        """
        self._version = 1

        self._nodes = []
        self._faces = []
        self._cells = []

        self._groups = {}

        self._generate(start_coord, end_coord, num)

    def _generate(self, start, end, num):
        # generate nodes
        dx = (end.x - start.x) / (num - 1)
        dy = (end.y - start.y) / (num - 1)
        dz = (end.z - start.z) / (num - 1)
        for i in range(num):
            x = start.x + i * dx
            y = start.y + i * dy
            z = start.z + i * dz
            node = Node(i, Coordinate(x, y, z))
            self._nodes.append(node)

        # generate mesh
        for i in range(num):
            node1 = self._nodes[i]
            normal = Vector(0, 1)

            # face
            face1 = Face(i, [i], node1.coord, 1, 1, normal)
            self._faces.append(face1)

            # cell
            if i == num - 1:
                break
            node2 = self._nodes[i + 1]
            length = abs(node2.coord.x - node1.coord.x)
            center = 0.5 * (node1.coord + node2.coord)
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

    def __init__(
        self, lower_left: Coordinate, upper_right: Coordinate, num_x: int, num_y: int
    ):
        """
        Initialize a 2D structured grid.

        Args:
            lower_left: The lower left corner of the grid.
            upper_right: The upper right corner of the grid.
            num_x: The number of nodes in the x-direction.
            num_y: The number of nodes in the y-direction.
        """
        self._ll = lower_left
        self._ur = upper_right
        self._nx = num_x
        self._ny = num_y

        self._nodes = []
        self._faces = []
        self._cells = []

        self._stats = {}
        self._groups = {}

        self._generate()
        self._statistics()

    def _generate(self):
        dx = (self._ur.x - self._ll.x) / self._nx
        dy = (self._ur.y - self._ll.y) / self._ny

        for i in range(self._nx):
            x = self._ll.x + i * dx
            for j in range(self._ny):
                y = self._ll.y + (j + 0.5) * dy
                node = Node(i * self._ny + j, Coordinate(x, y))
                self._nodes.append(node)

        for i in range(self._nx - 1):
            for j in range(self._ny - 1):
                n_lu = self._nodes[i * self._ny + j]
                n_ru = self._nodes[(i + 1) * self._ny + j]
                n_rl = self._nodes[(i + 1) * self._ny + j + 1]
                n_ll = self._nodes[i * self._ny + j + 1]

                # face 1, n_lu -> n_ru
                fid1 = i * self._ny + j
                nodes = [n_lu.id, n_ru.id]
                center = 0.5 * (n_lu.coord + n_ru.coord)
                perimeter = abs(n_ru.coord.x - n_lu.coord.x)
                area = perimeter
                normal = Vector(0, 1)
                face1 = Face(fid1, nodes, center, perimeter, area, normal)
                self._faces.append(face1)

                # face 2, n_ru -> n_rl
                fid2 = i * self._ny + j + 1
                nodes = [n_ru.id, n_rl.id]
                center = 0.5 * (n_ru.coord + n_rl.coord)
                perimeter = abs(n_rl.coord.x - n_ru.coord.x)
                area = perimeter
                normal = Vector(1, 0)
                face2 = Face(fid2, nodes, center, perimeter, area, normal)
                self._faces.append(face2)

                # face 3, n_rl -> n_ll
                fid3 = (i + 1) * self._ny + j
                nodes = [n_rl.id, n_ll.id]
                center = 0.5 * (n_rl.coord + n_ll.coord)
                perimeter = abs(n_ll.coord.x - n_rl.coord.x)
                area = perimeter
                normal = Vector(0, -1)
                face3 = Face(fid3, nodes, center, perimeter, area, normal)
                self._faces.append(face3)

                # face 4, n_ll -> n_lu
                fid4 = (i + 1) * self._ny + j + 1
                nodes = [n_ll.id, n_lu.id]
                center = 0.5 * (n_ll.coord + n_lu.coord)
                perimeter = abs(n_lu.coord.x - n_ll.coord.x)
                area = perimeter
                normal = Vector(-1, 0)
                face4 = Face(fid4, nodes, center, perimeter, area, normal)
                self._faces.append(face4)

                # cell
                cid = i * self._ny + j
                faces = [fid1, fid2, fid3, fid4]
                center = 0.25 * (
                    face1.center + face2.center + face3.center + face4.center
                )
                surface = dx * dy
                volume = surface
                cell = Cell(cid, faces, center, surface, volume)
                self._cells.append(cell)

    def _statistics(self):
        """
        Calculate and store some statistics of the grid.
        """
        self._stats["node"] = {
            "min_z": np.min([node.coord.z for node in self._nodes]),
            "max_z": np.max([node.coord.z for node in self._nodes]),
            "avg_z": np.mean([node.coord.z for node in self._nodes]),
        }

        self._stats["face"] = {
            "min_area": np.min([face.area for face in self._faces]),
            "max_area": np.max([face.area for face in self._faces]),
            "avg_area": np.mean([face.area for face in self._faces]),
        }

        self._stats["cell"] = {
            "min_volume": np.min([cell.volume for cell in self._cells]),
            "max_volume": np.max([cell.volume for cell in self._cells]),
            "avg_volume": np.mean([cell.volume for cell in self._cells]),
        }

    @property
    def version(self) -> int:
        return 1.0

    @property
    def domain(self) -> str:
        return "2d"

    @property
    def extents(self) -> list:
        return [self._ll.x, self._ur.x], [self._ll.y, self._ur.y], None

    @property
    def node_count(self) -> int:
        return len(self._nodes)

    @property
    def nodes(self) -> list:
        return self._nodes

    @property
    def stat_node_elevation(self) -> tuple:
        return self._stats["node"].values()

    @property
    def face_count(self) -> int:
        return len(self._faces)

    @property
    def faces(self) -> list:
        return self._faces

    @property
    def stat_face_area(self) -> tuple:
        return self._stats["face"].values()

    @property
    def cell_count(self) -> int:
        return len(self._cells)

    @property
    def cells(self) -> list:
        return self._cells

    @property
    def stat_cell_volume(self) -> tuple:
        return self._stats["cell"].values()

    def set_node_group(self, group_name: str, node_indices: list):
        for node_index in node_indices:
            if node_index >= self.node_count or node_index < 0:
                raise ValueError(f"Node index {node_index} out of range.")

        if group_name in self._groups:
            logger.warning(f"Group {group_name} already exists, overwriting.")

        self._groups[group_name] = node_indices

    def set_face_group(self, group_name: str, face_indices: list):
        for face_index in face_indices:
            if face_index >= self.face_count or face_index < 0:
                raise ValueError(f"Face index {face_index} out of range.")

        if group_name in self._groups:
            logger.warning(f"Group {group_name} already exists, overwriting.")

        self._groups[group_name] = face_indices

    def set_cell_group(self, group_name: str, cell_indices: list):
        for cell_index in cell_indices:
            if cell_index >= self.cell_count or cell_index < 0:
                raise ValueError(f"Cell index {cell_index} out of range.")

        if group_name in self._groups:
            logger.warning(f"Group {group_name} already exists, overwriting.")

        self._groups[group_name] = cell_indices

    def delete_group(self, group_name: str):
        if group_name in self._groups:
            self._groups.pop(group_name)

    def get_group(self, group_name: str) -> list:
        if group_name in self._groups:
            return self._groups[group_name]
        else:
            return None


class Grid3D(Mesh):
    """3D structured grid."""

    def __init__(self, start, end, num):
        pass

    @property
    def version(self) -> int:
        pass

    @property
    def domain(self) -> str:
        pass

    @property
    def extents(self) -> list:
        pass

    @property
    def node_count(self) -> int:
        pass

    @property
    def nodes(self) -> list:
        pass

    @property
    def stat_node_elevation(self) -> tuple:
        pass

    @property
    def face_count(self) -> int:
        pass

    @property
    def faces(self) -> list:
        pass

    @property
    def stat_face_area(self) -> tuple:
        pass

    @property
    def cell_count(self) -> int:
        pass

    @property
    def cells(self) -> list:
        pass

    @property
    def stat_cell_volume(self) -> tuple:
        pass

    # -----------------------------------------------
    # --- element modification methods ---
    # -----------------------------------------------

    def refine_cell(self, index: int):
        pass

    def relax_cell(self, index: int):
        pass

    # -----------------------------------------------
    # --- additional methods ---
    # -----------------------------------------------

    def set_node_group(self, group_name: str, node_indices: list):
        pass

    def set_face_group(self, group_name: str, face_indices: list):
        pass

    def set_cell_group(self, group_name: str, cell_indices: list):
        pass

    def delete_group(self, group_name: str):
        pass

    def get_group(self, group_name: str) -> list:
        pass
