# -*- encoding: utf-8 -*-
"""
Copyright (C) 2024, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

1d/2d/3d structured grids.
"""
from yunmeng.numerics.enums import MeshDimension
from yunmeng.numerics.mesh.elements import Coordinate, Node, Face, Cell
from yunmeng.numerics.mesh.spatials import Grid
from yunmeng.numerics.algos.topos import sort_anticlockwise, calculate_center
import numpy as np


# -----------------------------------------------
# region Grid2D
# -----------------------------------------------


class Grid2D(Grid):
    """2D structured grid in x-y plane (zero z-coordinate).

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

        # Horizontal Segments (Lying on y=const, connecting x_i and x_{i+1})
        # These serve as the South and North faces of the cells.
        h_face_count = (self._nx - 1) * self._ny
        h_faces = []

        # Iterate to match the logical index:
        # face_h(i, j) connects node(i,j) and node(i+1, j)
        for i in range(self._nx - 1):
            for j in range(self._ny):
                lid = i * self._ny + j
                rid = (i + 1) * self._ny + j
                node_ids = [lid, rid]

                center = 0.5 * (
                    self._nodes[lid].coordinate + self._nodes[rid].coordinate
                )
                h_faces.append(Face(center, node_ids))

        # Vertical Segments (Lying on x=const, connecting y_j and y_{j+1})
        # These serve as the West and East faces of the cells.
        v_face_count = self._nx * (self._ny - 1)
        v_faces = []

        # Storage order: i (0..Nx-1) outer, j (0..Ny-2) inner.
        for i in range(self._nx):
            for j in range(self._ny - 1):
                bid = i * self._ny + j
                tid = i * self._ny + (j + 1)
                node_ids = [bid, tid]

                center = 0.5 * (
                    self._nodes[bid].coordinate + self._nodes[tid].coordinate
                )
                v_faces.append(Face(center, node_ids))

        # Combine all faces
        self._faces = np.array(h_faces + v_faces)
        h_offset = 0
        v_offset = len(h_faces)

        # Generate Cells
        cell_size = (self._nx - 1) * (self._ny - 1)
        cells = []
        for i in range(self._nx - 1):
            for j in range(self._ny - 1):
                # Calculate global face indices
                idx_s = i * self._ny + j
                idx_n = i * self._ny + (j + 1)
                idx_w = v_offset + i * (self._ny - 1) + j
                idx_e = v_offset + (i + 1) * (self._ny - 1) + j
                face_ids = [idx_n, idx_w, idx_s, idx_e]

                # Retrieve face objects and sort
                faces = self.get_faces(face_ids)
                sorted_faces, sorted_face_ids = sort_anticlockwise(faces, face_ids)

                # Calculate cell center
                center = calculate_center(sorted_faces)
                cells.append(Cell(center, sorted_face_ids))

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
