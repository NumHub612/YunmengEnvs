# -*- encoding: utf-8 -*-
"""
Copyright (C) 2024, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

1d/2d/2d structured grids.
"""
from core.numerics.enums import MeshDimension
from core.numerics.mesh.elements import Coordinate, Node, Face, Cell
from core.numerics.mesh.spatials import Grid
from core.numerics.algos.topos import (
    sort_anticlockwise,
    calculate_center,
)


# -----------------------------------------------
# region --- Grid1D ---
# -----------------------------------------------


class Grid1D(Grid):
    """1D uniform structured grid in x-direction.

    Notes:
        - The 1d grid is a special case, somehow it's viered.
        - All nodes y- ans z-coordinates are set to 0.
    """

    def __init__(self, start: Coordinate, end: Coordinate, num: int):
        """
        Initialize a 1D uniform structured grid.

        Args:
            start_coord: The starting coordinate of the grid.
            end_coord: The ending coordinate of the grid.
            num: The number of nodes in the grid.
        """
        super().__init__()
        self._dim = MeshDimension.D1
        self._nx = num
        self._dx = (end.x - start.x) / (num - 1)

        self._generate(start, end, num)

    def _generate(self, start, end, num):
        # generate nodes
        for i in range(num):
            x = start.x + i * self._dx
            node = Node(i, Coordinate(x))
            self._nodes.append(node)

        # generate mesh
        for i in range(num):
            node1 = self._nodes[i]

            # face
            face1 = Face(i, node1.coordinate, [i])
            self._faces.append(face1)

            # cell
            if i == num - 1:
                break
            node2 = self._nodes[i + 1]
            center = 0.5 * (node1.coordinate + node2.coordinate)
            cell = Cell(i, center, [i, i + 1])
            self._cells.append(cell)

    def match_node(self, i: int, j: int = None, k: int = None) -> int:
        return i

    def match_cell(self, i: int, j: int = None, k: int = None) -> int:
        return i

    def retrieve_node_neighbours(self, index: int) -> list:
        east = index + 1 if index < self._nx - 1 else None
        west = index - 1 if index > 0 else None
        return [east, west, None, None, None, None]

    def retrieve_cell_neighbours(self, index: int) -> list:
        east = index + 1 if index < self._nx - 1 else None
        west = index - 1 if index > 0 else None
        return [east, west, None, None, None, None]


# -----------------------------------------------
# region --- Grid2D ---
# -----------------------------------------------


class Grid2D(Grid):
    """2D structured grid in x-y plane."""

    def __init__(
        self,
        lower_left: Coordinate,
        upper_right: Coordinate,
        num_x: int,
        num_y: int,
        mode: str = None,
        **kwargs
    ):
        """
        Initialize a 2D structured grid.

        Args:
            lower_left: The lower left corner of the grid.
            upper_right: The upper right corner.
            num_x: The number of nodes in the x-direction.
            num_y: The number of nodes in the y-direction.
            mode: The node distribution mode.
            kwargs: The extra settings corresponding to `mode`.

        Note:
            if `mode` is None, the extra settings would be invalid.
            if `mode` is xxx, the following configs needed:
                + dx (float):
                + dy (float):
            if `mode` is xxx: the following configs needed:
                + ratio_x (float):
                + ratio_x (float):
            if `mode` is xxx: the following configs needed:
                + pos_x (list):
                + pos_y (list):
        """
        super().__init__()
        self._dim = MeshDimension.D2
        self._ll = lower_left
        self._ur = upper_right
        self._nx = num_x
        self._ny = num_y
        self._dx = None
        self._dy = None
        self._dz = None

        self._generate()

    def _generate(self):
        self._dx = (self._ur.x - self._ll.x) / (self._nx - 1)
        self._dy = (self._ur.y - self._ll.y) / (self._ny - 1)

        # generate nodes
        nid = 0
        for i in range(self._nx):
            x = self._ll.x + i * self._dx
            for j in range(self._ny):
                y = self._ll.y + j * self._dy
                node = Node(nid, Coordinate(x, y))
                self._nodes.append(node)
                nid += 1

        # generate faces
        fid = 0
        for i in range(self._nx):
            for j in range(self._ny):
                n_lu = self._nodes[i * self._ny + j]

                # face 1, n_lu -> n_ru
                if i < self._nx - 1:
                    n_ru = self._nodes[(i + 1) * self._ny + j]
                    nodes = sorted([n_lu.id, n_ru.id], reverse=True)
                    center = 0.5 * (n_lu.coordinate + n_ru.coordinate)
                    face1 = Face(fid, center, nodes)
                    self._faces.append(face1)
                    fid += 1

                # face 2, n_lu -> n_ld
                if j < self._ny - 1:
                    n_ld = self._nodes[i * self._ny + j + 1]
                    nodes = sorted([n_lu.id, n_ld.id])
                    center = 0.5 * (n_lu.coordinate + n_ld.coordinate)
                    face2 = Face(fid, center, nodes)
                    self._faces.append(face2)
                    fid += 1

        # generate cells
        cid = 0
        for i in range(self._nx - 1):
            for j in range(self._ny - 1):
                f_n = i * (2 * (self._ny - 1) + 1) + 2 * j
                f_w = f_n + 1
                f_s = f_w + 1
                if i < self._nx - 2:
                    f_e = (i + 1) * (2 * (self._ny - 1) + 1) + 2 * j + 1
                else:
                    f_e = (i + 1) * (2 * (self._ny - 1) + 1) + j

                face_ids = [f_n, f_w, f_s, f_e]
                faces = self.get_faces(face_ids)
                faces = sort_anticlockwise(faces)
                center = calculate_center(faces)
                face_ids = [f.id for f in faces]
                cell = Cell(cid, center, face_ids)
                self._cells.append(cell)
                cid += 1

    def match_node(self, i: int, j: int, k: int = None) -> int:
        if i < 0 or i >= self._nx or j < 0 or j >= self._ny:
            return None

        nid = i * self._ny + j
        return nid if 0 <= nid < self.node_count else None

    def match_cell(self, i: int, j: int, k: int = None) -> int:
        if i < 0 or i >= self._nx - 1 or j < 0 or j >= self._ny - 1:
            return None

        cid = i * (self._ny - 1) + j
        return cid if 0 <= cid < self.cell_count else None

    def retrieve_node_neighbours(self, index: int) -> list:
        i = index // self._ny
        j = index % self._ny

        north = self.match_node(i, j + 1)
        south = self.match_node(i, j - 1)
        west = self.match_node(i - 1, j)
        east = self.match_node(i + 1, j)
        return [east, west, north, south, None, None]

    def retrieve_cell_neighbours(self, index: int) -> list:
        i = index // (self._ny - 1)
        j = index % (self._ny - 1)

        north = self.match_cell(i, j + 1)
        south = self.match_cell(i, j - 1)
        west = self.match_cell(i - 1, j)
        east = self.match_cell(i + 1, j)
        return [east, west, north, south, None, None]


# -----------------------------------------------
# region --- Grid3D ---
# -----------------------------------------------


class Grid3D(Grid):
    """3D structured grid."""

    def __init__(
        self,
        lower_left_front: Coordinate,
        upper_right_back: Coordinate,
        num_x: int,
        num_y: int,
        num_z: int,
    ):
        """
        Initialize a 3D structured grid.

        Args:
            lower_left_front: The lower left front corner of the grid.
            upper_right_back: The upper right back corner of the grid.
            num_x: The number of nodes in the x-direction.
            num_y: The number of nodes in the y-direction.
            num_z: The number of nodes in the z-direction.
        """
        super().__init__()
        self._dim = MeshDimension.D3
        self._ll = lower_left_front
        self._ur = upper_right_back
        self._nx = num_x
        self._ny = num_y
        self._nz = num_z
        self._dx = None
        self._dy = None
        self._dz = None

        self._generate()

    def _generate(self):
        self._dx = (self._ur.x - self._ll.x) / (self._nx - 1)
        self._dy = (self._ur.y - self._ll.y) / (self._ny - 1)
        self._dz = (self._ur.z - self._ll.z) / (self._nz - 1)

        # generate nodes
        nid = 0
        for k in range(self._nz):
            z = self._ll.z + k * self._dz
            for j in range(self._ny):
                y = self._ll.y + j * self._dy
                for i in range(self._nx):
                    x = self._ll.x + i * self._dx
                    node = Node(nid, Coordinate(x, y, z))
                    self._nodes.append(node)
                    nid += 1

        # generate faces
        fid = 0
        for k in range(self._nz):
            # faces in x-direction
            for j in range(self._ny - 1):
                if k >= self._nz - 1:
                    continue
                for i in range(self._nx):
                    n_ll = k * self._nx * self._ny + j * self._nx + i
                    n_rl = k * self._nx * self._ny + (j + 1) * self._nx + i
                    n_ru = (k + 1) * self._nx * self._ny + (j + 1) * self._nx + i
                    n_lu = (k + 1) * self._nx * self._ny + j * self._nx + i
                    node_ids = [n_ll, n_rl, n_ru, n_lu]
                    nodes = self.get_nodes(node_ids)
                    nodes = sort_anticlockwise(nodes)
                    center = calculate_center(nodes)
                    node_ids = [n.id for n in nodes]
                    face = Face(fid, center, node_ids)
                    self._faces.append(face)
                    fid += 1

            # faces in y-direction
            for i in range(self._nx - 1):
                if k >= self._nz - 1:
                    continue
                for j in range(self._ny):
                    n_rl = k * self._nx * self._ny + j * self._nx + i
                    n_ru = (k + 1) * self._nx * self._ny + j * self._nx + i
                    n_lu = (k + 1) * self._nx * self._ny + j * self._nx + i + 1
                    n_ll = k * self._nx * self._ny + j * self._nx + i + 1
                    node_ids = [n_rl, n_ru, n_lu, n_ll]
                    nodes = self.get_nodes(node_ids)
                    nodes = sort_anticlockwise(nodes)
                    center = calculate_center(nodes)
                    node_ids = [n.id for n in nodes]
                    face = Face(fid, center, node_ids)
                    self._faces.append(face)
                    fid += 1

            # faces in z-direction
            for j in range(self._ny - 1):
                for i in range(self._nx - 1):
                    n_lu = k * self._nx * self._ny + j * self._nx + i
                    n_ll = k * self._nx * self._ny + j * self._nx + i + 1
                    n_ul = k * self._nx * self._ny + (j + 1) * self._nx + i + 1
                    n_lr = k * self._nx * self._ny + (j + 1) * self._nx + i
                    node_ids = [n_lu, n_ll, n_ul, n_lr]
                    nodes = self.get_nodes(node_ids)
                    nodes = sort_anticlockwise(nodes)
                    center = calculate_center(nodes)
                    node_ids = [n.id for n in nodes]
                    face = Face(fid, center, node_ids)
                    self._faces.append(face)
                    fid += 1

        # generate cells
        cid = 0
        faces_along_x = self._nx * (self._ny - 1)
        faces_along_y = self._ny * (self._nx - 1)
        faces_along_z = (self._nx - 1) * (self._ny - 1)
        faces_per_layer = (
            (self._nx - 1) * (self._ny - 1)
            + self._nx * (self._ny - 1)
            + self._ny * (self._nx - 1)
        )
        for k in range(self._nz - 1):
            for j in range(self._ny - 1):
                for i in range(self._nx - 1):
                    f_n = k * faces_per_layer + j * self._nx + i
                    f_s = k * faces_per_layer + j * self._nx + i + 1
                    f_w = k * faces_per_layer + faces_along_x + i * self._ny + j
                    f_e = k * faces_per_layer + faces_along_x + i * self._ny + j + 1
                    f_d = (
                        k * faces_per_layer
                        + faces_along_x
                        + faces_along_y
                        + j * (self._nx - 1)
                        + i
                    )
                    if k < self._nz - 2:
                        f_u = (
                            (k + 1) * faces_per_layer
                            + faces_along_x
                            + faces_along_y
                            + j * (self._nx - 1)
                            + i
                        )
                    else:
                        f_u = (
                            k * faces_per_layer
                            + faces_along_z
                            + faces_along_x
                            + faces_along_y
                            + j * (self._nx - 1)
                            + i
                        )
                    face_ids = [f_n, f_s, f_w, f_e, f_d, f_u]
                    faces = self.get_faces(face_ids)
                    center = calculate_center(faces)
                    cell = Cell(cid, center, face_ids)
                    self._cells.append(cell)
                    cid += 1

    def match_node(self, i: int, j: int, k: int) -> int:
        if i < 0 or i >= self._nx or j < 0 or j >= self._ny or k < 0 or k >= self._nz:
            return None

        return k * self._nx * self._ny + j * self._nx + i

    def match_cell(self, i: int, j: int, k: int) -> int:
        if (
            i < 0
            or i >= self._nx - 1
            or j < 0
            or j >= self._ny - 1
            or k < 0
            or k >= self._nz - 1
        ):
            return None

        return k * (self._nx - 1) * (self._ny - 1) + j * (self._nx - 1) + i

    def retrieve_node_neighbours(self, index: int) -> list:
        k = index // (self._nx * self._ny)
        j = (index - k * self._nx * self._ny) // self._nx
        i = index % self._nx

        north = self.match_node(i, j + 1, k)
        south = self.match_node(i, j - 1, k)
        west = self.match_node(i - 1, j, k)
        east = self.match_node(i + 1, j, k)
        down = self.match_node(i, j, k - 1)
        up = self.match_node(i, j, k + 1)
        return [east, west, north, south, up, down]

    def retrieve_cell_neighbours(self, index: int) -> list:
        k = index // ((self._nx - 1) * (self._ny - 1))
        j = (index - k * (self._nx - 1) * (self._ny - 1)) // (self._nx - 1)
        i = (index - k * (self._nx - 1) * (self._ny - 1)) % (self._nx - 1)

        north = self.match_cell(i, j + 1, k)
        south = self.match_cell(i, j - 1, k)
        west = self.match_cell(i - 1, j, k)
        east = self.match_cell(i + 1, j, k)
        down = self.match_cell(i, j, k - 1)
        up = self.match_cell(i, j, k + 1)
        return [east, west, north, south, up, down]


# -----------------------------------------------
# region --- QuadGrid2D ---
# -----------------------------------------------


class QuadGrid2D(Grid2D):
    """2D structured grid with quadrilateral cells."""

    def __init__(self):
        pass
