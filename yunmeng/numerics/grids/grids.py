# -*- encoding: utf-8 -*-
"""
Copyright (C) 2024, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

1d/2d/3d structured grids.
"""

from yunmeng.numerics.enums import MeshDimension
from yunmeng.numerics.mesh import Coordinate, Node, Face, Cell
from yunmeng.numerics.mesh import sort_anticlockwise, calculate_center
from yunmeng.numerics.grids.grid import Grid
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

        # --- Precomputed neighbor index arrays ---
        self._node_neigh_e: np.ndarray = None
        self._node_neigh_w: np.ndarray = None
        self._node_neigh_n: np.ndarray = None
        self._node_neigh_s: np.ndarray = None

        # Generate mesh
        self._generate()

        # Build neighbor index arrays
        self._build_node_neighbour_indices()

    # -----------------------------------------------
    # region Constructors
    # -----------------------------------------------

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

    # -----------------------------------------------
    # region Adjacency
    # -----------------------------------------------

    def _build_node_neighbour_indices(self):
        """Precompute neighbor global indices for ALL nodes."""
        n_nodes = self.node_count
        self._node_neigh_e = np.empty(n_nodes, dtype=np.int64)
        self._node_neigh_w = np.empty(n_nodes, dtype=np.int64)
        self._node_neigh_n = np.empty(n_nodes, dtype=np.int64)
        self._node_neigh_s = np.empty(n_nodes, dtype=np.int64)

        # Fill with -1 (indicating None/out of bounds)
        self._node_neigh_e.fill(-1)
        self._node_neigh_w.fill(-1)
        self._node_neigh_n.fill(-1)
        self._node_neigh_s.fill(-1)

        # Structured grid formula: nid = i * ny + j
        # East:  (i+1, j) -> nid + ny
        # West:  (i-1, j) -> nid - ny
        # North: (i, j+1) -> nid + 1
        # South: (i, j-1) -> nid - 1
        ny = self._ny
        for i in range(self._nx):
            for j in range(self._ny):
                nid = i * ny + j
                if i + 1 < self._nx:
                    self._node_neigh_e[nid] = (i + 1) * ny + j
                if i > 0:
                    self._node_neigh_w[nid] = (i - 1) * ny + j
                if j + 1 < self._ny:
                    self._node_neigh_n[nid] = i * ny + (j + 1)
                if j > 0:
                    self._node_neigh_s[nid] = i * ny + (j - 1)

    def _neigh_arr_to_val(self, arr: np.ndarray, nid: int):
        """Convert array lookup to int or None."""
        v = arr[nid]
        return int(v) if v >= 0 else None

    def get_node_neighbours(self, index: int) -> list:
        """Get the neighbours node indices."""
        if self._node_neigh_e is None:
            # Fallback: on-demand calc (should not happen)
            i = index // self._ny
            j = index % self._ny
            north = self.match_node(i, j + 1)
            south = self.match_node(i, j - 1)
            west = self.match_node(i - 1, j)
            east = self.match_node(i + 1, j)
            return [east, west, north, south, None, None]

        return [
            self._neigh_arr_to_val(self._node_neigh_e, index),
            self._neigh_arr_to_val(self._node_neigh_w, index),
            self._neigh_arr_to_val(self._node_neigh_n, index),
            self._neigh_arr_to_val(self._node_neigh_s, index),
            None,
            None,
        ]

    def get_node_neighbours_batch(
        self, indices: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Bulk neighbour query for vectorized operators.

        Returns east, west, north, south arrays (all shape same as indices).
        Invalid neighbors are marked with -1.
        """
        return (
            self._node_neigh_e[indices],
            self._node_neigh_w[indices],
            self._node_neigh_n[indices],
            self._node_neigh_s[indices],
        )

    def get_node_neighbours_all(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return the full precomputed neighbor index arrays."""
        return (
            self._node_neigh_e,
            self._node_neigh_w,
            self._node_neigh_n,
            self._node_neigh_s,
        )

    # -----------------------------------------------
    # region Indexing
    # -----------------------------------------------

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

    def get_cell_neighbours(self, index: int) -> list:
        i = index // (self._ny - 1)
        j = index % (self._ny - 1)

        north = self.match_cell(i, j + 1)
        south = self.match_cell(i, j - 1)
        west = self.match_cell(i - 1, j)
        east = self.match_cell(i + 1, j)
        return [east, west, north, south, None, None]
