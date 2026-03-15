# -*- encoding: utf-8 -*-
"""
Copyright (C) 2024, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

1d/2d/3d structured grids.
"""
from core.numerics.enums import MeshDimension
from core.numerics.mesh.elements import Coordinate, Node, Face, Cell
from core.numerics.mesh.spatials import Grid
from core.numerics.algos.topos import sort_anticlockwise, calculate_center
import numpy as np


# -----------------------------------------------
# region  Grid1D
# -----------------------------------------------


# -----------------------------------------------
# region Grid2D
# -----------------------------------------------


class Grid2D(Grid):
    """2D structured grid in x-y plane.

    NOTE:
    The grid's index is encoded by columns from the bottom left corner.
    """

    def __init__(self, x_positions: np.ndarray, y_positions: np.ndarray):
        """
        Initialize a 2D structured grid with explicit node positions.

        Args:
            x_positions: Array of x positions for the grid nodes
            y_positions: Array of y positions for the grid nodes
        """
        super().__init__()
        self._dim = MeshDimension.D2

        x_positions = np.asarray(x_positions)
        y_positions = np.asarray(y_positions)

        # Validate inputs
        if len(x_positions) < 2:
            raise ValueError("x_positions must more than 2 points")
        if len(y_positions) < 2:
            raise ValueError("y_positions must more than 2 points")
        if not np.all(np.diff(x_positions) > 0):
            raise ValueError("x_positions must be increasing")
        if not np.all(np.diff(y_positions) > 0):
            raise ValueError("y_positions must be increasing")

        # Store positions
        self._x_positions = x_positions
        self._y_positions = y_positions

        # Calculate grid dimensions
        self._nx = len(x_positions)
        self._ny = len(y_positions)

        # Calculate lengths
        self._lx = abs(x_positions[-1] - x_positions[0])
        self._ly = abs(y_positions[-1] - y_positions[0])

        # Calculate average spacings
        self._dx = self._lx / (self._nx - 1)
        self._dy = self._ly / (self._ny - 1)

        # Determine if grid is uniform
        self._uniform = np.allclose(np.diff(x_positions), self._dx) and np.allclose(
            np.diff(y_positions), self._dy
        )

        # Generate mesh
        self._generate()

    @staticmethod
    def by_uniform(
        lower_left: Coordinate, upper_right: Coordinate, num_x: int, num_y: int
    ) -> "Grid2D":
        """
        Create a uniform 2D grid.

        Args:
            lower_left: The lower left corner of the grid
            upper_right: The upper right corner
            num_x: Number of nodes in the x-axis
            num_y: Number of nodes in the y-axis
        """
        x_positions = np.linspace(lower_left.x, upper_right.x, num_x)
        y_positions = np.linspace(lower_left.y, upper_right.y, num_y)
        return Grid2D(x_positions, y_positions)

    @staticmethod
    def by_custom(
        lower_left: Coordinate, upper_right: Coordinate, xs: list, ys: list
    ) -> "Grid2D":
        """
        Create a 2D grid with custom node positions.

        Args:
            lower_left: The lower left corner of the grid
            upper_right: The upper right corner
            xs: List of x positions
            ys: List of y positions
        """
        lx = upper_right.x - lower_left.x
        ly = upper_right.y - lower_left.y

        # Normalize and scale x positions
        x_positions = np.array(xs)
        x_positions = (x_positions - x_positions[0]) / (
            x_positions[-1] - x_positions[0]
        ) * lx + lower_left.x

        # Normalize and scale y positions
        y_positions = np.array(ys)
        y_positions = (y_positions - y_positions[0]) / (
            y_positions[-1] - y_positions[0]
        ) * ly + lower_left.y

        return Grid2D(x_positions, y_positions)

    def _generate(self):
        """Generate the grid."""
        xs = self._x_positions
        ys = self._y_positions
        node_size = self._nx * self._ny

        # Set dx and dy
        if self._uniform:
            self._dx = xs[1] - xs[0]
            self._dy = ys[1] - ys[0]
        else:
            self._dx = self._lx / (self._nx - 1)
            self._dy = self._ly / (self._ny - 1)

        # Create meshgrid
        X, Y = np.meshgrid(xs, ys, indexing="ij")

        # Generate coordinates
        coords = np.column_stack((X.ravel(), Y.ravel(), np.zeros(node_size)))

        # Create nodes
        nodes = np.array(
            [Node(Coordinate(x, y, 0.0)) for x, y in zip(coords[:, 0], coords[:, 1])]
        )
        self._nodes = nodes

        # Generate faces using vectorized operations
        # Horizontal faces (x-direction)
        h_face_count = (self._nx - 1) * self._ny
        h_node_ids = np.arange(h_face_count * 2).reshape(-1, 2)

        # Calculate node indices for horizontal faces
        i_indices = np.repeat(np.arange(self._nx - 1), self._ny)
        j_indices = np.tile(np.arange(self._ny), self._nx - 1)

        # Left and right node IDs for horizontal faces
        left_ids = i_indices * self._ny + j_indices
        right_ids = (i_indices + 1) * self._ny + j_indices

        # Create horizontal faces
        h_faces = []
        for lid, rid in zip(left_ids, right_ids):
            node_ids = sorted([lid, rid], reverse=True)
            center = 0.5 * (self._nodes[lid].coordinate + self._nodes[rid].coordinate)
            h_faces.append(Face(center, node_ids))

        # Vertical faces (y-direction)
        v_face_count = self._nx * (self._ny - 1)
        i_indices = np.repeat(np.arange(self._nx), self._ny - 1)
        j_indices = np.tile(np.arange(self._ny - 1), self._nx)

        # Bottom and top node IDs for vertical faces
        bottom_ids = i_indices * self._ny + j_indices
        top_ids = i_indices * self._ny + (j_indices + 1)

        # Create vertical faces
        v_faces = []
        for bid, tid in zip(bottom_ids, top_ids):
            node_ids = sorted([bid, tid])
            center = 0.5 * (self._nodes[bid].coordinate + self._nodes[tid].coordinate)
            v_faces.append(Face(center, node_ids))

        # Combine all faces
        self._faces = np.array(h_faces + v_faces)

        # Generate cells using vectorized operations
        cell_size = (self._nx - 1) * (self._ny - 1)

        # Calculate face indices for each cell
        # North face index
        i_indices = np.repeat(np.arange(self._nx - 1), self._ny - 1)
        j_indices = np.tile(np.arange(self._ny - 1), self._nx - 1)
        f_n = i_indices * (2 * (self._ny - 1) + 1) + 2 * j_indices

        # West face index
        f_w = f_n + 1

        # South face index
        f_s = f_w + 1

        # East face index (depends on i)
        mask = i_indices < self._nx - 2
        f_e = np.where(
            mask,
            (i_indices + 1) * (2 * (self._ny - 1) + 1) + 2 * j_indices + 1,
            (i_indices + 1) * (2 * (self._ny - 1) + 1) + j_indices,
        )

        # Create cells
        cells = []
        for idx in range(cell_size):
            face_ids = [f_n[idx], f_w[idx], f_s[idx], f_e[idx]]
            faces = self.get_faces(face_ids)
            faces, face_ids = sort_anticlockwise(faces, face_ids)
            center = calculate_center(faces)
            cells.append(Cell(center, face_ids))

        self._cells = np.array(cells)

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

    def get_node_neighbours(self, index: int) -> list:
        i = index // self._ny
        j = index % self._ny

        north = self.match_node(i, j + 1)
        south = self.match_node(i, j - 1)
        west = self.match_node(i - 1, j)
        east = self.match_node(i + 1, j)
        return [east, west, north, south, None, None]

    def get_cell_neighbours(self, index: int) -> list:
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
