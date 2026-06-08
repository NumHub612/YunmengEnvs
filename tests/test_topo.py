# -*- encoding: utf-8 -*-
"""
Unit tests for MeshTopo
"""

import pytest
import numpy as np
from yunmeng.numerics.mesh.grids import Grid2D, Grid
from yunmeng.numerics.mesh.meshes import GenericMesh, Mesh
from yunmeng.numerics.mesh.elements import Coordinate

# ============================================
# region Fixtures
# ============================================


@pytest.fixture
def grid2d():
    """Create a 4x4 Grid2D mesh (16 nodes, 24 faces, 9 cells)"""
    lower_left = Coordinate(0.0, 0.0, 0.0)
    upper_right = Coordinate(3.0, 3.0, 0.0)
    return Grid2D.by_uniform(lower_left, upper_right, 4, 4)


@pytest.fixture
def mesh2d():
    """2D unstructured mesh with 32 triangular cells
    (4x4 grid divided into triangles)."""
    # Create a 4x4 grid of nodes
    nodes = []
    for j in range(5):
        for i in range(5):
            nodes.append([float(i), float(j), 0.0])

    faces = []
    # Horizontal edges (0 to 19)
    for j in range(5):
        for i in range(4):
            faces.append([j * 5 + i, j * 5 + i + 1])

    # Vertical edges (20 to 39)
    for j in range(4):
        for i in range(5):
            faces.append([j * 5 + i, (j + 1) * 5 + i])

    cells = []
    for j in range(4):
        for i in range(4):
            n0 = j * 5 + i  # Bottom-Left
            n1 = j * 5 + i + 1  # Bottom-Right
            n2 = (j + 1) * 5 + i  # Top-Left
            n3 = (j + 1) * 5 + i + 1  # Top-Right

            # Calculate Face Indices based on the generation order above
            bottom_face = j * 4 + i
            top_face = (j + 1) * 4 + i
            left_face = 20 + j * 5 + i
            right_face = 20 + j * 5 + (i + 1)

            # Diagonal face index: 40 + current_cell_index
            diagonal_face = 40 + (j * 4 + i)

            # Add the diagonal edge to faces list: connects n0 and n3
            faces.append([n0, n3])

            # Triangle 1: Bottom-Right half (Nodes: n0, n1, n3)
            cells.append([bottom_face, right_face, diagonal_face])

            # Triangle 2: Top-Left half (Nodes: n0, n3, n2)
            cells.append([diagonal_face, top_face, left_face])

    return GenericMesh(nodes, faces, cells)


# ============================================
# region Grid2D
# ============================================


class TestGrid2DTopo:
    """Test Grid2D mesh topology"""

    def test_boundary_detection(self, grid2d: Grid):
        """Test boundary detection for 4x4 Grid2D mesh"""
        topo = grid2d.get_topo_assistant()

        # 4x4 mesh: 12 boundary nodes, 4 internal nodes
        assert len(topo.boundary_nodes) == 12
        assert len(topo.internal_nodes) == 4

        # 12 boundary faces, 12 internal faces
        assert len(topo.boundary_faces) == 12
        assert len(topo.internal_faces) == 12

        # 4 boundary cells, 5 internal cells
        assert len(topo.boundary_cells) == 8
        assert len(topo.internal_cells) == 1

    def test_cell_neighbours(self, grid2d: Grid):
        """Test cell neighbor relationships for 4x4 Grid2D mesh"""
        topo = grid2d.get_topo_assistant()
        neighbours = topo.cell_neighbours

        # Corner cell (0) should have 2 neighbors
        assert len(neighbours[0]) == 2
        assert set(neighbours[0]) == {1, 3}

        # Boundary cell (1) should have 3 neighbors
        assert len(neighbours[1]) == 3
        assert set(neighbours[1]) == {0, 2, 4}

        # Internal cell (4) should have 4 neighbors
        assert len(neighbours[4]) == 4
        assert set(neighbours[4]) == {1, 3, 5, 7}

    def test_face_cells(self, grid2d: Grid):
        """Test face-cell relationships for 4x4 Grid2D mesh"""
        topo = grid2d.get_topo_assistant()
        face_cells = topo.face_cells

        # Boundary faces should be connected to only one cell
        for face_id in topo.boundary_faces:
            c1, c2 = face_cells[face_id]
            assert (c1 is not None and c2 is None) or (c1 is None and c2 is not None)

        # Internal faces should be connected to two cells
        for face_id in topo.internal_faces:
            c1, c2 = face_cells[face_id]
            assert c1 is not None and c2 is not None

    def test_cell_faces(self, grid2d: Grid):
        """Test cell-face relationships for 4x4 Grid2D mesh"""
        topo = grid2d.get_topo_assistant()
        cell_faces = topo.cell_faces

        # Each cell should have 4 faces
        for faces in cell_faces:
            assert len(faces) == 4

    def test_cell_nodes(self, grid2d: Grid):
        """Test cell-node relationships for 4x4 Grid2D mesh"""
        topo = grid2d.get_topo_assistant()
        cell_nodes = topo.cell_nodes

        # Each cell should have 4 nodes
        for nodes in cell_nodes:
            assert len(nodes) == 4

    def test_node_faces(self, grid2d: Grid):
        """Test node-face relationships for 4x4 Grid2D mesh"""
        topo = grid2d.get_topo_assistant()
        node_faces = topo.node_faces

        # Boundary nodes should be connected to 2-3 faces
        for node_id in topo.boundary_nodes:
            assert 2 <= len(node_faces[node_id]) <= 3

        # Internal nodes should be connected to 4 faces
        for node_id in topo.internal_nodes:
            assert len(node_faces[node_id]) == 4

    def test_node_cells(self, grid2d: Grid):
        """Test node-cell relationships for 4x4 Grid2D mesh"""
        topo = grid2d.get_topo_assistant()
        node_cells = topo.node_cells

        # Boundary nodes should be connected to 1-2 cells
        for node_id in topo.boundary_nodes:
            assert 1 <= len(node_cells[node_id]) <= 2

        # Internal nodes should be connected to 4 cells
        for node_id in topo.internal_nodes:
            assert len(node_cells[node_id]) == 4

    def test_grid_properties(self, grid2d: Grid):
        """Test grid properties"""
        grid = grid2d

        # Test grid dimension
        assert grid.dimension is not None

        # Test element counts
        assert grid.node_count == 16
        assert grid.face_count == 24
        assert grid.cell_count == 9

        # Test orthogonality
        assert grid.orthogonal is True

        # Test version
        assert grid.version >= 1

    def test_topo_cache_mechanism(self, grid2d: Grid):
        """Test topology cache mechanism"""
        topo = grid2d.get_topo_assistant()

        # Multiple accesses should return the same result
        boundary_nodes_1 = topo.boundary_nodes
        boundary_nodes_2 = topo.boundary_nodes
        np.testing.assert_array_equal(boundary_nodes_1, boundary_nodes_2)

        # Test reset method
        topo.reset(grid2d)
        boundary_nodes_3 = topo.boundary_nodes
        np.testing.assert_array_equal(boundary_nodes_1, boundary_nodes_3)


# ============================================
# region GenericMesh
# ============================================


class TestMesh2DTopo:
    """Test GenericMesh topology for triangular mesh"""

    def test_boundary_detection(self, mesh2d: Mesh):
        """Test boundary detection for triangular GenericMesh"""
        topo = mesh2d.get_topo_assistant()

        # grid with triangular cells: 16 boundary nodes, 4 internal nodes
        assert len(topo.boundary_nodes) == 16
        assert len(topo.internal_nodes) == 9

        # For triangular mesh: 16 boundary faces, 40 internal faces
        assert len(topo.boundary_faces) == 16
        assert len(topo.internal_faces) == 40

        # 16 triangular cells: all are boundary cells except the center ones
        assert len(topo.boundary_cells) == 14
        assert len(topo.internal_cells) == 18

    def test_face_cells_relation(self, mesh2d: Mesh):
        """Test face-cell relationships for triangular GenericMesh"""
        topo = mesh2d.get_topo_assistant()
        face_cells = topo.face_cells

        # Boundary faces should be connected to only one cell
        assert face_cells[0][0] is not None
        assert face_cells[0][1] is None

        # Internal faces (diagonal edges) should be connected to two cells
        assert face_cells[40][0] is not None
        assert face_cells[40][1] is not None

    def test_cell_neighbours(self, mesh2d: Mesh):
        """Test cell neighbor relationships for triangular GenericMesh"""
        topo = mesh2d.get_topo_assistant()
        neighbours = topo.cell_neighbours

        # Corner triangular cell
        assert np.all(neighbours[0] == [1, 3])

        # Edge triangular cell
        assert np.all(neighbours[2] == [3, 5])

        # Center triangular cells
        assert np.all(neighbours[10] == [11, 3, 13])

    def test_cell_faces(self, mesh2d: Mesh):
        """Test cell-face relationships for triangular GenericMesh"""
        topo = mesh2d.get_topo_assistant()
        cell_faces = topo.cell_faces

        # Each triangular cell should have 3 faces
        for faces in cell_faces:
            assert len(faces) == 3

    def test_cell_nodes(self, mesh2d: Mesh):
        """Test cell-node relationships for triangular GenericMesh"""
        topo = mesh2d.get_topo_assistant()
        cell_nodes = topo.cell_nodes

        # Each triangular cell should have 3 nodes
        for nodes in cell_nodes:
            assert len(nodes) == 3

    def test_node_faces(self, mesh2d: Mesh):
        """Test node-face relationships for triangular GenericMesh"""
        topo = mesh2d.get_topo_assistant()
        node_faces = topo.node_faces

        # Corner nodes
        assert len(node_faces[0]) == 3
        assert len(node_faces[20]) == 2

        # Edge nodes
        assert len(node_faces[21]) == 4
        assert len(node_faces[2]) == 4

        # Internal nodes
        assert len(node_faces[16]) == 6
        assert len(node_faces[7]) == 6

    def test_node_cells(self, mesh2d: Mesh):
        """Test node-cell relationships for triangular GenericMesh"""
        topo = mesh2d.get_topo_assistant()
        node_cells = topo.node_cells

        # Corner nodes
        assert len(node_cells[0]) == 2
        assert len(node_cells[20]) == 1

        # Edge nodes
        assert len(node_cells[1]) == 3
        assert len(node_cells[22]) == 3

        # Internal nodes
        assert len(node_cells[6]) == 6
        assert len(node_cells[17]) == 6

    def test_mesh_properties(self, mesh2d: Mesh):
        """Test mesh properties"""
        mesh = mesh2d

        # Test mesh dimension
        assert mesh.dimension is not None

        # Test element counts for triangular mesh
        assert mesh.node_count == 25
        assert mesh.face_count == 56
        assert mesh.cell_count == 32

        # Test orthogonality
        assert mesh.orthogonal == False

        # Test version
        assert mesh.version >= 1

    def test_topo_cache_mechanism(self, mesh2d: Mesh):
        """Test topology cache mechanism"""
        topo = mesh2d.get_topo_assistant()

        # Multiple accesses should return the same result
        boundary_nodes_1 = topo.boundary_nodes
        boundary_nodes_2 = topo.boundary_nodes
        np.testing.assert_array_equal(boundary_nodes_1, boundary_nodes_2)

        # Test reset method
        topo.reset(mesh2d)
        boundary_nodes_3 = topo.boundary_nodes
        np.testing.assert_array_equal(boundary_nodes_1, boundary_nodes_3)

    def test_face_nodes_anticlockwise(self, mesh2d: Mesh):
        """Test that face nodes are sorted in anticlockwise order"""
        topo = mesh2d.get_topo_assistant()
        face_nodes = topo.face_nodes

        # Check that each face has 2 nodes (edges)
        for nodes in face_nodes:
            assert len(nodes) == 2

    def test_node_neighbours(self, mesh2d: Mesh):
        """Test node neighbor relationships for triangular GenericMesh"""
        topo = mesh2d.get_topo_assistant()
        node_neighbours = topo.node_neighbours

        # Corner nodes
        assert len(node_neighbours[0]) == 3
        assert len(node_neighbours[20]) == 2

        # Edge nodes
        assert len(node_neighbours[1]) == 4

        # Internal nodes
        assert len(node_neighbours[11]) == 6
