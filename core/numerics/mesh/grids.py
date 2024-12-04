# -*- encoding: utf-8 -*-
"""
Copyright (C) 2024, The YunmengEnvs Contributors. Join us, for you talents!  

1d/2d/2d structured grids.
"""
from core.numerics.mesh import Mesh, MeshGeom, Coordinate, Node, Face, Cell
from core.numerics.fields import Vector
from configs.settings import logger

import numpy as np
from abc import abstractmethod


class Grid(Mesh):
    """Base class for orthogonal structured grids."""

    def __init__(self):
        super().__init__()

    # -----------------------------------------------
    # --- Extenal grid properties ---
    # -----------------------------------------------

    @property
    @abstractmethod
    def nx(self) -> int:
        """The number of discretization size in the x-direction."""
        pass

    @property
    @abstractmethod
    def ny(self) -> int:
        """The number of discretization size in the y-direction."""
        pass

    @property
    @abstractmethod
    def nz(self) -> int:
        """The number of discretization size in the z-direction."""
        pass

    # -----------------------------------------------
    # --- Extenal grid methods ---
    # -----------------------------------------------

    @abstractmethod
    def match_node(self, i: int, j: int, k: int) -> int:
        """
        Match the global node index with the local indexes.

        Args:
            i: The local index in the x-direction.
            j: The local index in the y-direction.
            k: The local index in the z-direction.

        Returns:
            The global node index.
        """
        pass

    @abstractmethod
    def match_cell(self, i: int, j: int, k: int) -> int:
        """
        Match the global cell index with the local indexes.

        Args:
            i: The local index in the x-direction.
            j: The local index in the y-direction.
            k: The local index in the z-direction.

        Returns:
            The global cell index.
        """
        pass

    @abstractmethod
    def retrieve_node_neighborhoods(self, index: int) -> tuple:
        """
        Get the neighborhood node indexes of the given node.

        Args:
            index: The global node index at center.

        Returns:
            The neighborhood node global indexes sorted in the following order:
                - the north neighbor index.
                - the south neighbor index.
                - the east neighbor index.
                - the west neighbor index.
                - the top neighbor index.
                - the bottom neighbor index.
        """
        pass

    @abstractmethod
    def retrieve_cell_neighborhoods(self, index: int) -> tuple:
        """
        Get the neighborhood cell indexes of the given cell.

        Args:
            index: The global cell index at center.

        Returns:
            The neighborhood cell global indexes sorted in the following order:
                - the north neighbor index.
                - the south neighbor index.
                - the east neighbor index.
                - the west neighbor index.
                - the top neighbor index.
                - the bottom neighbor index.
        """
        pass


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
        self._nx = num

        self._generate(start, end, num)

    def _generate(self, start, end, num):
        # generate nodes
        dx = (end.x - start.x) / (num - 1)
        for i in range(num):
            x = start.x + i * dx
            node = Node(i, Coordinate(x))
            self._nodes.append(node)

        # generate mesh
        for i in range(num):
            node1 = self._nodes[i]
            normal = Vector(0, 1)

            # face
            face1 = Face(i, [i], node1.coordinate, 1, 1, normal)
            self._faces.append(face1)

            # cell
            if i == num - 1:
                break
            node2 = self._nodes[i + 1]
            length = abs(node2.coordinate.x - node1.coordinate.x)
            center = 0.5 * (node1.coordinate + node2.coordinate)
            cell = Cell(i, [i, i + 1], center, length, length)
            self._cells.append(cell)

    @property
    def domain(self) -> str:
        return "1d"

    @property
    def nx(self) -> int:
        return self._nx

    @property
    def ny(self) -> int:
        return None

    @property
    def nz(self) -> int:
        return None

    def refine_cell(self, index: int):
        pass

    def relax_cell(self, index: int):
        pass

    def match_node(self, i: int, j: int = None, k: int = None) -> int:
        return i

    def match_cell(self, i: int, j: int = None, k: int = None) -> int:
        return i

    def retrieve_node_neighborhoods(self, index: int) -> tuple:
        east = index + 1 if index < self._nx - 1 else None
        west = index - 1 if index > 0 else None
        return None, None, east, west, None, None

    def retrieve_cell_neighborhoods(self, index: int) -> tuple:
        east = index + 1 if index < self._nx - 1 else None
        west = index - 1 if index > 0 else None
        return None, None, east, west, None, None


class Grid2D(Grid):
    """2D structured grid in x-y plane.

    Notes:
        - All nodes z-coordinate are set to 0.
    """

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
        super().__init__()
        self._ll = lower_left
        self._ur = upper_right
        self._nx = num_x
        self._ny = num_y

        self._generate()

    def _generate(self):
        dx = (self._ur.x - self._ll.x) / (self._nx - 1)
        dy = (self._ur.y - self._ll.y) / (self._ny - 1)

        # generate nodes
        nid = 0
        for i in range(self._nx):
            x = self._ll.x + i * dx
            for j in range(self._ny):
                y = self._ll.y + j * dy
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
                    nodes = [n_lu.id, n_ru.id]
                    center = 0.5 * (n_lu.coordinate + n_ru.coordinate)
                    perimeter = abs(n_ru.coordinate.x - n_lu.coordinate.x)
                    area = perimeter
                    normal = Vector(0, 1)
                    face1 = Face(fid, nodes, center, perimeter, area, normal)
                    self._faces.append(face1)
                    fid += 1

                # face 2, n_lu -> n_ld
                if j < self._ny - 1:
                    n_ld = self._nodes[i * self._ny + j + 1]
                    nodes = [n_lu.id, n_ld.id]
                    center = 0.5 * (n_lu.coordinate + n_ld.coordinate)
                    perimeter = abs(n_ld.coordinate.y - n_lu.coordinate.y)
                    area = perimeter
                    normal = Vector(1, 0)
                    face2 = Face(fid, nodes, center, perimeter, area, normal)
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

                faces = [f_n, f_w, f_s, f_e]
                center = MeshGeom.calculate_center(
                    [self._faces[id].coordinate for id in faces]
                )
                surface = dx * dy
                volume = surface
                cell = Cell(cid, faces, center, surface, volume)
                self._cells.append(cell)
                cid += 1

    @property
    def domain(self) -> str:
        return "2d"

    @property
    def nx(self) -> int:
        return self._nx

    @property
    def ny(self) -> int:
        return self._ny

    @property
    def nz(self) -> int:
        return None

    def refine_cell(self, index: int):
        pass

    def relax_cell(self, index: int):
        pass

    def match_node(self, i: int, j: int, k: int = None) -> int:
        return i * self._ny + j

    def match_cell(self, i: int, j: int, k: int = None) -> int:
        return i * (self._ny - 1) + j

    def retrieve_node_neighborhoods(self, index: int) -> tuple:
        i = index // self._ny
        j = index % self._ny

        north = i * self._ny + j - 1 if j > 0 else None
        south = i * self._ny + j + 1 if j < self._ny - 1 else None
        west = (i - 1) * self._ny + j if i > 0 else None
        east = (i + 1) * self._ny + j if i < self._nx - 1 else None
        return north, south, east, west, None, None

    def retrieve_cell_neighborhoods(self, index: int) -> tuple:
        i = index // (self._ny - 1)
        j = index % (self._ny - 1)

        north = i * (self._ny - 1) + j - 1 if j > 0 else None
        south = i * (self._ny - 1) + j + 1 if j < self._ny - 2 else None
        west = (i - 1) * (self._ny - 1) + j if i > 0 else None
        east = (i + 1) * (self._ny - 1) + j if i < self._nx - 2 else None
        return north, south, east, west, None, None


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
        self._ll = lower_left_front
        self._ur = upper_right_back
        self._nx = num_x
        self._ny = num_y
        self._nz = num_z

        self._generate()

    def _generate(self):
        dx = (self._ur.x - self._ll.x) / (self._nx - 1)
        dy = (self._ur.y - self._ll.y) / (self._ny - 1)
        dz = (self._ur.z - self._ll.z) / (self._nz - 1)

        # generate nodes
        nid = 0
        for k in range(self._nz):
            z = self._ll.z + k * dz
            for j in range(self._ny):
                y = self._ll.y + j * dy
                for i in range(self._nx):
                    x = self._ll.x + i * dx
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
                    nodes = [n_ll, n_rl, n_ru, n_lu]

                    center = MeshGeom.calculate_center(
                        [self._nodes[id].coordinate for id in nodes]
                    )
                    perimeter = 2 * dy + 2 * dz
                    area = dy * dz
                    normal = Vector(1, 0, 0)
                    face = Face(fid, nodes, center, perimeter, area, normal)
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
                    nodes = [n_rl, n_ru, n_lu, n_ll]

                    center = MeshGeom.calculate_center(
                        [self._nodes[id].coordinate for id in nodes]
                    )
                    perimeter = 2 * dx + 2 * dz
                    area = dx * dz
                    normal = Vector(0, 1, 0)
                    face = Face(fid, nodes, center, perimeter, area, normal)
                    self._faces.append(face)
                    fid += 1

            # faces in z-direction
            for j in range(self._ny - 1):
                for i in range(self._nx - 1):
                    n_lu = k * self._nx * self._ny + j * self._nx + i
                    n_ll = k * self._nx * self._ny + j * self._nx + i + 1
                    n_ul = k * self._nx * self._ny + (j + 1) * self._nx + i + 1
                    n_lr = k * self._nx * self._ny + (j + 1) * self._nx + i
                    nodes = [n_lu, n_ll, n_ul, n_lr]

                    center = MeshGeom.calculate_center(
                        [self._nodes[id].coordinate for id in nodes]
                    )
                    perimeter = 2 * dx + 2 * dy
                    area = dx * dy
                    normal = Vector(0, 0, 1)
                    face = Face(fid, nodes, center, perimeter, area, normal)
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
                    faces = [f_n, f_s, f_w, f_e, f_d, f_u]

                    center = MeshGeom.calculate_center(
                        [self._faces[id].coordinate for id in faces]
                    )
                    surface = 2 * dx * dy + 2 * dy * dz + 2 * dx * dz
                    volume = dx * dy * dz
                    cell = Cell(cid, faces, center, surface, volume)
                    self._cells.append(cell)
                    cid += 1

    @property
    def domain(self) -> str:
        return "3d"

    @property
    def nx(self) -> int:
        return self._nx

    @property
    def ny(self) -> int:
        return self._ny

    @property
    def nz(self) -> int:
        return None

    def refine_cell(self, index: int):
        pass

    def relax_cell(self, index: int):
        pass

    def match_node(self, i: int, j: int, k: int) -> int:
        return k * self._nx * self._ny + j * self._nx + i

    def match_cell(self, i: int, j: int, k: int) -> int:
        return k * (self._nx - 1) * (self._ny - 1) + j * (self._nx - 1) + i

    def retrieve_node_neighborhoods(self, index: int) -> tuple:
        k = index // (self._nx * self._ny)
        j = (index - k * self._nx * self._ny) // self._nx
        i = index % self._nx

        north = k * self._nx * self._ny + (j - 1) * self._nx + i if j > 0 else None
        south = (
            k * self._nx * self._ny + (j + 1) * self._nx + i
            if j < self._ny - 1
            else None
        )
        west = k * self._nx * self._ny + j * self._nx + i - 1 if i > 0 else None
        east = (
            k * self._nx * self._ny + j * self._nx + i + 1 if i < self._nx - 1 else None
        )
        down = (k - 1) * self._nx * self._ny + j * self._nx + i if k > 0 else None
        up = (
            (k + 1) * self._nx * self._ny + j * self._nx + i
            if k < self._nz - 1
            else None
        )
        return north, south, east, west, down, up

    def retrieve_cell_neighborhoods(self, index: int) -> tuple:
        k = index // ((self._nx - 1) * (self._ny - 1))
        j = (index - k * (self._nx - 1) * (self._ny - 1)) // (self._nx - 1)
        i = (index - k * (self._nx - 1) * (self._ny - 1)) % (self._nx - 1)

        north = (
            k * (self._nx - 1) * (self._ny - 1) + (j - 1) * (self._nx - 1) + i
            if j > 0
            else None
        )
        south = (
            k * (self._nx - 1) * (self._ny - 1) + (j + 1) * (self._nx - 1) + i
            if j < self._ny - 2
            else None
        )
        west = (
            k * (self._nx - 1) * (self._ny - 1) + j * (self._nx - 1) + i - 1
            if i > 0
            else None
        )
        east = (
            k * (self._nx - 1) * (self._ny - 1) + j * (self._nx - 1) + i + 1
            if i < self._nx - 2
            else None
        )
        down = (
            k * (self._nx - 1) * (self._ny - 1) + (j - 1) * (self._nx - 1) + i
            if k > 0
            else None
        )
        up = (
            k * (self._nx - 1) * (self._ny - 1) + (j + 1) * (self._nx - 1) + i
            if k < self._nz - 2
            else None
        )
        return north, south, east, west, down, up
