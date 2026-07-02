# -*- encoding: utf-8 -*-
"""
Tests for Grid2D class.
"""

import numpy as np
from yunmeng.numerics.grids import Grid2D
from yunmeng.numerics.mesh import Coordinate
from yunmeng.numerics.enums import MeshDimension


class TestGrid2D:
    """Test cases for Grid2D class."""

    def test_init_with_positions(self):
        """Test Grid2D initialization with explicit positions."""
        x_positions = np.array([0.0, 1.0, 2.0])
        y_positions = np.array([0.0, 1.0, 2.0])

        grid = Grid2D(x_positions, y_positions)

        assert grid.nx == 3
        assert grid.ny == 3
        assert grid.node_count == 9
        assert grid.cell_count == 4
        assert grid.dimension == MeshDimension.D2

    def test_uniform_grid(self):
        """Test creating a uniform grid."""
        lower_left = Coordinate(0.0, 0.0, 0.0)
        upper_right = Coordinate(2.0, 2.0, 0.0)
        num_x = 3
        num_y = 3

        grid = Grid2D.by_uniform(lower_left, upper_right, num_x, num_y)

        assert grid.nx == num_x
        assert grid.ny == num_y
        assert grid.node_count == num_x * num_y
        assert grid.cell_count == (num_x - 1) * (num_y - 1)
        assert grid.uniform == True

    def test_custom_grid(self):
        """Test creating a custom grid."""
        lower_left = Coordinate(0.0, 0.0, 0.0)
        upper_right = Coordinate(2.0, 2.0, 0.0)
        pos_x = [0.0, 0.5, 1.5, 2.0]
        pos_y = [0.0, 0.5, 1.5, 2.0]

        grid = Grid2D.by_custom(lower_left, upper_right, pos_x, pos_y)

        assert grid.nx == len(pos_x)
        assert grid.ny == len(pos_y)
        assert grid.uniform == False

    def test_node_access(self):
        """Test accessing nodes."""
        x_positions = np.array([0.0, 1.0, 2.0])
        y_positions = np.array([0.0, 1.0, 2.0])

        grid = Grid2D(x_positions, y_positions)

        # Test getting nodes
        assert grid.node_count == 9
        assert grid.get_nodes([0]) is not None
        assert grid.get_nodes([8]) is not None

        # Test matching nodes
        assert grid.match_node(0, 0) == 0
        assert grid.match_node(1, 0) == 3
        assert grid.match_node(0, 1) == 1
        assert grid.match_node(1, 1) == 4
        assert grid.match_node(2, 2) == 8

        # Test out of bounds
        assert grid.match_node(-1, 0) is None
        assert grid.match_node(3, 0) is None
        assert grid.match_node(0, -1) is None
        assert grid.match_node(0, 3) is None

    def test_cell_access(self):
        """Test accessing cells."""
        x_positions = np.array([0.0, 1.0, 2.0])
        y_positions = np.array([0.0, 1.0, 2.0])

        grid = Grid2D(x_positions, y_positions)

        # Test getting cells
        assert grid.cell_count == 4
        assert grid.get_cells([0]) is not None
        assert grid.get_cells([3]) is not None

        # Test matching cells
        assert grid.match_cell(0, 0) == 0
        assert grid.match_cell(1, 0) == 2
        assert grid.match_cell(0, 1) == 1
        assert grid.match_cell(1, 1) == 3

        # Test out of bounds
        assert grid.match_cell(-1, 0) is None
        assert grid.match_cell(2, 0) is None
        assert grid.match_cell(0, -1) is None
        assert grid.match_cell(0, 2) is None

    def test_face_access(self):
        """Test accessing faces."""
        x_positions = np.array([0.0, 1.0, 2.0])
        y_positions = np.array([0.0, 1.0, 2.0])

        grid = Grid2D(x_positions, y_positions)

        # Test getting faces
        assert grid.face_count == 12  # (3-1)*3 + 3*(3-1) = 6 + 6
        assert grid.get_faces([0]) is not None
        assert grid.get_faces([11]) is not None

    def test_node_neighbours(self):
        """Test getting node neighbours."""
        x_positions = np.array([0.0, 1.0, 2.0])
        y_positions = np.array([0.0, 1.0, 2.0])

        grid = Grid2D(x_positions, y_positions)

        # Test interior node (index 4, at position (1, 1))
        neighbours = grid.get_node_neighbours(4)
        assert neighbours[0] == 7  # east
        assert neighbours[1] == 1  # west
        assert neighbours[2] == 5  # north
        assert neighbours[3] == 3  # south
        assert neighbours[4] is None  # up
        assert neighbours[5] is None  # down

        # Test boundary node (index 0, at position (0, 0))
        neighbours = grid.get_node_neighbours(0)
        assert neighbours[0] == 3  # east
        assert neighbours[1] is None  # west
        assert neighbours[2] == 1  # north
        assert neighbours[3] is None  # south

    def test_cell_neighbours(self):
        """Test getting cell neighbours."""
        x_positions = np.array([0.0, 1.0, 2.0])
        y_positions = np.array([0.0, 1.0, 2.0])

        grid = Grid2D(x_positions, y_positions)

        # Test interior cell (index 0, at position (0, 0))
        neighbours = grid.get_cell_neighbours(0)
        assert neighbours[0] == 2  # east
        assert neighbours[1] is None  # west
        assert neighbours[2] == 1  # north
        assert neighbours[3] is None  # south

    def test_grid_properties(self):
        """Test grid properties."""
        x_positions = np.array([0.0, 1.0, 2.0])
        y_positions = np.array([0.0, 1.0, 2.0])

        grid = Grid2D(x_positions, y_positions)

        # Test lengths
        assert np.allclose(grid.lx, 2.0)
        assert np.allclose(grid.ly, 2.0)

        # Test dimensions
        assert grid.nx == 3
        assert grid.ny == 3

    def test_non_uniform_grid(self):
        """Test non-uniform grid properties."""
        x_positions = np.array([0.0, 0.5, 2.0])
        y_positions = np.array([0.0, 0.5, 2.0])

        grid = Grid2D(x_positions, y_positions)
        assert grid.uniform == False
