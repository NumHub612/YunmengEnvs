# -*- encoding: utf-8 -*-
"""
Unit tests for MeshTopo
"""

import pytest
import numpy as np
from core.numerics.mesh.grids import Grid2D
from core.numerics.mesh.meshes import GenericMesh, Mesh
from core.numerics.mesh.elements import Coordinate


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
    """Create a 4x4 GenericMesh (16 nodes, 24 faces, 9 cells)"""
    nodes = [
        [0.0, 0.0, 0.0],  # 0
        [1.0, 0.0, 0.0],  # 1
        [2.0, 0.0, 0.0],  # 2
        [3.0, 0.0, 0.0],  # 3
        [0.0, 1.0, 0.0],  # 4
        [1.0, 1.0, 0.0],  # 5
        [2.0, 1.0, 0.0],  # 6
        [3.0, 1.0, 0.0],  # 7
        [0.0, 2.0, 0.0],  # 8
        [1.0, 2.0, 0.0],  # 9
        [2.0, 2.0, 0.0],  # 10
        [3.0, 2.0, 0.0],  # 11
        [0.0, 3.0, 0.0],  # 12
        [1.0, 3.0, 0.0],  # 13
        [2.0, 3.0, 0.0],  # 14
        [3.0, 3.0, 0.0],  # 15
    ]
    faces = [
        [0, 1],  # 0
        [1, 2],  # 1
        [2, 3],  # 2
        [4, 5],  # 3
        [5, 6],  # 4
        [6, 7],  # 5
        [8, 9],  # 6
        [9, 10],  # 7
        [10, 11],  # 8
        [12, 13],  # 9
        [13, 14],  # 10
        [14, 15],  # 11
        [0, 4],  # 12
        [4, 8],  # 13
        [8, 12],  # 14
        [1, 5],  # 15
        [5, 9],  # 16
        [9, 13],  # 17
        [2, 6],  # 18
        [6, 10],  # 19
        [10, 14],  # 20
        [3, 7],  # 21
        [7, 11],  # 22
        [11, 15],  # 23
    ]
    cells = [
        [0, 12, 3, 15],  # 0
        [1, 15, 4, 18],  # 1
        [2, 18, 5, 21],  # 2
        [3, 13, 6, 16],  # 3
        [4, 16, 7, 19],  # 4
        [5, 19, 8, 22],  # 5
        [6, 14, 9, 17],  # 6
        [7, 17, 10, 20],  # 7
        [8, 20, 11, 23],  # 8
    ]
    return GenericMesh(nodes, faces, cells)


# ============================================
# region Grid2D
# ============================================


class TestGrid2DTopo:
    """Test Grid2D mesh topology"""

    def test_boundary_detection_4x4(self, grid2d: Grid2D):
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

    def test_cell_neighbours_4x4(self, grid2d: Grid2D):
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

    def test_face_cells_4x4(self, grid2d: Grid2D):
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

    def test_cell_faces_4x4(self, grid2d: Grid2D):
        """Test cell-face relationships for 4x4 Grid2D mesh"""
        topo = grid2d.get_topo_assistant()
        cell_faces = topo.cell_faces

        # Each cell should have 4 faces
        for faces in cell_faces:
            assert len(faces) == 4

    def test_cell_nodes_4x4(self, grid2d: Grid2D):
        """Test cell-node relationships for 4x4 Grid2D mesh"""
        topo = grid2d.get_topo_assistant()
        cell_nodes = topo.cell_nodes

        # Each cell should have 4 nodes
        for nodes in cell_nodes:
            assert len(nodes) == 4

    def test_node_faces_4x4(self, grid2d: Grid2D):
        """Test node-face relationships for 4x4 Grid2D mesh"""
        topo = grid2d.get_topo_assistant()
        node_faces = topo.node_faces

        # Boundary nodes should be connected to 2-3 faces
        for node_id in topo.boundary_nodes:
            assert 2 <= len(node_faces[node_id]) <= 3

        # Internal nodes should be connected to 4 faces
        print(len(node_faces))
        for node_id in topo.internal_nodes:
            print(node_id)
            assert len(node_faces[node_id]) == 4

    def test_node_cells_4x4(self, grid2d: Grid2D):
        """Test node-cell relationships for 4x4 Grid2D mesh"""
        topo = grid2d.get_topo_assistant()
        node_cells = topo.node_cells

        # Boundary nodes should be connected to 1-2 cells
        for node_id in topo.boundary_nodes:
            assert 1 <= len(node_cells[node_id]) <= 2

        # Internal nodes should be connected to 4 cells
        for node_id in topo.internal_nodes:
            assert len(node_cells[node_id]) == 4

    def test_grid_properties(self, grid2d: Grid2D):
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

    def test_topo_cache_mechanism(self, grid2d: Grid2D):
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
    """Test GenericMesh topology"""

    def test_boundary_detection_4x4(self, mesh2d: Mesh):
        """Test boundary detection for 4x4 GenericMesh"""
        topo = mesh2d.get_topo_assistant()

        # 12 boundary nodes, 4 internal nodes
        assert len(topo.boundary_nodes) == 12
        assert len(topo.internal_nodes) == 4

        # 12 boundary faces, 12 internal faces
        assert len(topo.boundary_faces) == 12
        assert len(topo.internal_faces) == 12

        # 4 boundary cells, 5 internal cells
        assert len(topo.boundary_cells) == 8
        assert len(topo.internal_cells) == 1

    def test_face_cells_relation(self, mesh2d: Mesh):
        """Test face-cell relationships for GenericMesh"""
        topo = mesh2d.get_topo_assistant()
        face_cells = topo.face_cells

        # Boundary faces should be connected to only one cell
        assert face_cells[0][0] == 0
        assert face_cells[0][1] is None

        # Internal faces should be connected to two cells
        assert face_cells[15][0] == 0
        assert face_cells[15][1] == 1

    def test_cell_neighbours(self, mesh2d: Mesh):
        """Test cell neighbor relationships for GenericMesh"""
        topo = mesh2d.get_topo_assistant()
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

    def test_cell_faces(self, mesh2d: Mesh):
        """Test cell-face relationships for GenericMesh"""
        topo = mesh2d.get_topo_assistant()
        cell_faces = topo.cell_faces

        # Each cell should have 4 faces
        for faces in cell_faces:
            assert len(faces) == 4

    def test_cell_nodes(self, mesh2d: Mesh):
        """Test cell-node relationships for GenericMesh"""
        topo = mesh2d.get_topo_assistant()
        cell_nodes = topo.cell_nodes

        # Each cell should have 4 nodes
        for nodes in cell_nodes:
            assert len(nodes) == 4

    def test_node_faces(self, mesh2d: Mesh):
        """Test node-face relationships for GenericMesh"""
        topo = mesh2d.get_topo_assistant()
        node_faces = topo.node_faces

        # Boundary nodes should be connected to 2-3 faces
        for node_id in topo.boundary_nodes:
            assert 2 <= len(node_faces[node_id]) <= 3

        # Internal nodes should be connected to 4 faces
        for node_id in topo.internal_nodes:
            assert len(node_faces[node_id]) == 4

    def test_node_cells(self, mesh2d: Mesh):
        """Test node-cell relationships for GenericMesh"""
        topo = mesh2d.get_topo_assistant()
        node_cells = topo.node_cells

        # Boundary nodes should be connected to 1-2 cells
        for node_id in topo.boundary_nodes:
            assert 1 <= len(node_cells[node_id]) <= 2

        # Internal nodes should be connected to 4 cells
        for node_id in topo.internal_nodes:
            assert len(node_cells[node_id]) == 4

    def test_mesh_properties(self, mesh2d: Mesh):
        """Test mesh properties"""
        mesh = mesh2d

        # Test mesh dimension
        assert mesh.dimension is not None

        # Test element counts
        assert mesh.node_count == 16
        assert mesh.face_count == 24
        assert mesh.cell_count == 9

        # Test orthogonality
        assert mesh.orthogonal is not None

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
