# -*- encoding: utf-8 -*-
"""
Unit tests for MeshGeom.
"""

import pytest
import numpy as np
from yunmeng.numerics.grids import Grid2D, Grid
from yunmeng.numerics.mesh import GenericMesh, Mesh, Coordinate

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


class TestGrid2DGeom:
    """Test Grid2D mesh geometry"""

    def test_face_perimeter(self, grid2d: Grid):
        """Test face perimeter for 4x4 Grid2D mesh"""
        geom = grid2d.get_geom_assistant()
        face_perimeters = geom.face_perimeter

        # In a 4x4 grid with uniform spacing of 1.0,
        # each face should have perimeter 1.0
        assert len(face_perimeters) == 24
        np.testing.assert_allclose(face_perimeters, np.ones(24), rtol=1e-10)

    def test_face_area(self, grid2d: Grid):
        """Test face area for 4x4 Grid2D mesh"""
        geom = grid2d.get_geom_assistant()
        face_areas = geom.face_area

        # In 2D, face area equals face perimeter
        face_perimeters = geom.face_perimeter
        np.testing.assert_allclose(face_areas, face_perimeters, rtol=1e-10)

    def test_face_normal(self, grid2d: Grid):
        """Test face normal vectors for 4x4 Grid2D mesh"""
        geom = grid2d.get_geom_assistant()
        face_normals = geom.face_normal

        # Each normal should be a unit vector
        for normal in face_normals:
            norm = np.linalg.norm(normal.to_numpy())
            assert abs(norm - 1.0) < 1e-10

        # Horizontal faces should have normals in y-direction
        # Faces 0-2, 3-5, 6-8, 9-11 are horizontal
        for i in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]:
            normal = face_normals[i].to_numpy()
            assert abs(normal[0]) < 1e-10
            assert abs(normal[2]) < 1e-10

        # Vertical faces should have normals in x-direction
        # Faces 12-14, 15-17, 18-20, 21-23 are vertical
        for i in [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]:
            normal = face_normals[i].to_numpy()
            assert abs(normal[1]) < 1e-10
            assert abs(normal[2]) < 1e-10

    def test_cell_volume(self, grid2d: Grid):
        """Test cell volume (area in 2D) for 4x4 Grid2D mesh"""
        geom = grid2d.get_geom_assistant()
        cell_volumes = geom.cell_volume

        # In a 4x4 grid with uniform spacing of 1.0,
        # each cell should have volume(area) of 1.0
        assert len(cell_volumes) == 9
        np.testing.assert_allclose(cell_volumes, np.ones(9), rtol=1e-10)

    def test_cell_surface(self, grid2d: Grid):
        """Test cell surface (perimeter in 2D) for 4x4 Grid2D mesh"""
        geom = grid2d.get_geom_assistant()
        cell_surfaces = geom.cell_surface

        # Each cell should have surface (perimeter) of 4.0
        assert len(cell_surfaces) == 9
        np.testing.assert_allclose(cell_surfaces, 4.0 * np.ones(9), rtol=1e-10)

    def test_cell2cell_distance(self, grid2d: Grid):
        """Test cell-to-cell distances for 4x4 Grid2D mesh"""
        geom = grid2d.get_geom_assistant()
        cell2cell_dists = geom.cell2cell_distance

        # Check distances between neighboring cells
        # Cell 0 and cell 1 should be 1.0 apart
        assert abs(cell2cell_dists[0][1] - 1.0) < 1e-10
        # Cell 0 and cell 3 should be 1.0 apart
        assert abs(cell2cell_dists[0][3] - 1.0) < 1e-10
        # Cell 4 and cell 7 should be 1.0 apart
        assert abs(cell2cell_dists[4][7] - 1.0) < 1e-10

    def test_cell2face_distance(self, grid2d: Grid):
        """Test cell-to-face distances for 4x4 Grid2D mesh"""
        geom = grid2d.get_geom_assistant()
        cell2face_dists = geom.cell2face_distance

        # Check distances between cells and their faces
        # For a uniform grid, each cell should be 0.5 away from each of its faces
        for cell_id in range(grid2d.cell_count):
            for face_id, dist in cell2face_dists[cell_id].items():
                assert abs(dist - 0.5) < 1e-10

    def test_cell2cell_vector(self, grid2d: Grid):
        """Test cell-to-cell vectors for 4x4 Grid2D mesh"""
        geom = grid2d.get_geom_assistant()
        cell2cell_vects = geom.cell2cell_vector

        # Check vectors between neighboring cells
        # Cell 0 to cell 1 should be in positive y-direction
        vec_0_1 = cell2cell_vects[0][1].to_numpy()
        assert abs(vec_0_1[0]) < 1e-10
        assert abs(vec_0_1[1] - 1.0) < 1e-10
        assert abs(vec_0_1[2]) < 1e-10

        # Cell 0 to cell 3 should be in positive x-direction
        vec_0_3 = cell2cell_vects[0][3].to_numpy()
        assert abs(vec_0_3[0] - 1.0) < 1e-10
        assert abs(vec_0_3[1]) < 1e-10
        assert abs(vec_0_3[2]) < 1e-10

    def test_cell2face_vector(self, grid2d: Grid):
        """Test cell-to-face vectors for 4x4 Grid2D mesh"""
        geom = grid2d.get_geom_assistant()
        cell2face_vects = geom.cell2face_vector

        # Check vectors from cell 0 to its faces
        # Cell 0 to face 1 should be in positive y-direction
        vec_0_1 = cell2face_vects[0][1].to_numpy()
        assert abs(vec_0_1[0]) < 1e-10
        assert abs(vec_0_1[1] - 1.0) < 1e-10
        assert abs(vec_0_1[2]) < 1e-10

        # Cell 0 to face 12 should be in negative x-direction
        vec_0_12 = cell2face_vects[0][12].to_numpy()
        assert abs(vec_0_12[0] + 1.0) < 1e-10
        assert abs(vec_0_12[1]) < 1e-10
        assert abs(vec_0_12[2]) < 1e-10

    def test_geom_cache_mechanism(self, grid2d: Grid):
        """Test geometry cache mechanism"""
        geom = grid2d.get_geom_assistant()

        # Multiple accesses should return the same result
        face_areas_1 = geom.face_area
        face_areas_2 = geom.face_area
        np.testing.assert_array_equal(face_areas_1, face_areas_2)

        # Test reset method
        geom.reset(grid2d)
        face_areas_3 = geom.face_area
        np.testing.assert_array_equal(face_areas_1, face_areas_3)


# ============================================
# region GenericMesh
# ============================================


class TestMesh2DGeom:
    """Test unstructured mesh geometry using mesh2d fixture"""

    def test_face_perimeter(self, mesh2d: Mesh):
        """Test face perimeter for unstructured mesh with triangular cells"""
        geom = mesh2d.get_geom_assistant()
        face_perimeters = geom.face_perimeter

        # The mesh has 56 faces: 40 horizontal/vertical edges + 16 diagonal edges
        # Horizontal and vertical edges should have perimeter 1.0
        # Diagonal edges should have perimeter sqrt(2)
        assert len(face_perimeters) == 56

        # First 40 faces are horizontal/vertical edges with perimeter 1.0
        np.testing.assert_allclose(face_perimeters[:40], np.ones(40), rtol=1e-10)

        # Last 16 faces are diagonal edges with perimeter sqrt(2)
        np.testing.assert_allclose(
            face_perimeters[40:], np.sqrt(2) * np.ones(16), rtol=1e-10
        )

    def test_face_area(self, mesh2d: Mesh):
        """Test face area for unstructured mesh"""
        geom = mesh2d.get_geom_assistant()
        face_areas = geom.face_area

        # In 2D, face area equals face perimeter
        face_perimeters = geom.face_perimeter
        np.testing.assert_allclose(face_areas, face_perimeters, rtol=1e-10)

    def test_face_normal(self, mesh2d: Mesh):
        """Test face normal vectors for unstructured mesh"""
        geom = mesh2d.get_geom_assistant()
        face_normals = geom.face_normal

        # Each normal should be a unit vector
        for normal in face_normals:
            norm = np.linalg.norm(normal.to_numpy())
            assert abs(norm - 1.0) < 1e-10

        # Face 0 should be in negative y-direction
        normal_0 = face_normals[0].to_numpy()
        assert np.allclose(normal_0, np.array([0.0, 1.0, 0.0]), rtol=1e-10)

        # Face 21 should be in positive x-direction
        normal_21 = face_normals[21].to_numpy()
        assert np.allclose(normal_21, np.array([1.0, 0.0, 0.0]), rtol=1e-10)

        # Face 40 should be in east-south direction
        normal_40 = face_normals[40].to_numpy()
        assert np.allclose(
            normal_40, np.array([np.sqrt(2) / 2, -np.sqrt(2) / 2, 0.0]), rtol=1e-10
        )

    def test_cell_volume(self, mesh2d: Mesh):
        """Test cell volume (area in 2D) for unstructured mesh"""
        geom = mesh2d.get_geom_assistant()
        cell_volumes = geom.cell_volume

        # The mesh has 32 triangular cells (2 per grid cell)
        # Each triangle should have area 0.5
        assert len(cell_volumes) == 32
        np.testing.assert_allclose(cell_volumes, 0.5 * np.ones(32), rtol=1e-10)

    def test_cell_surface(self, mesh2d: Mesh):
        """Test cell surface (perimeter in 2D) for unstructured mesh"""
        geom = mesh2d.get_geom_assistant()
        cell_surfaces = geom.cell_surface

        # Each triangular cell should have perimeter 2 + sqrt(2)
        assert len(cell_surfaces) == 32
        expected_perimeter = 2.0 + np.sqrt(2)
        np.testing.assert_allclose(
            cell_surfaces, expected_perimeter * np.ones(32), rtol=1e-10
        )

    def test_cell2cell_distance(self, mesh2d: Mesh):
        """Test cell-to-cell distances for unstructured mesh"""
        geom = mesh2d.get_geom_assistant()
        cell2cell_dists = geom.cell2cell_distance

        # Check distances between neighboring cells
        assert abs(cell2cell_dists[0][1] - np.sqrt(2) / 3) < 1e-10
        assert abs(cell2cell_dists[0][3] - np.sqrt(5) / 3) < 1e-10

    def test_cell2face_distance(self, mesh2d: Mesh):
        """Test cell-to-face distances for unstructured mesh"""
        geom = mesh2d.get_geom_assistant()
        cell2face_dists = geom.cell2face_distance

        # Check distances between cells and their faces
        assert abs(cell2face_dists[0][40] - np.sqrt(2) / 6) < 1e-10
        assert abs(cell2face_dists[0][0] - np.sqrt(5) / 6) < 1e-10
        assert abs(cell2face_dists[0][21] - np.sqrt(5) / 6) < 1e-10

    def test_cell2cell_vector(self, mesh2d: Mesh):
        """Test cell-to-cell vectors for unstructured mesh"""
        geom = mesh2d.get_geom_assistant()
        cell2cell_vects = geom.cell2cell_vector

        # Check vectors between neighboring cells
        vec_0_1 = cell2cell_vects[0][1].to_numpy()
        magnitude = np.sqrt(2) / 3
        assert np.allclose(
            vec_0_1, np.array([-1 / 3 / magnitude, 1 / 3 / magnitude, 0.0]), rtol=1e-10
        )

        magnitude = np.sqrt(5) / 3
        vec_0_3 = cell2cell_vects[0][3].to_numpy()
        assert np.allclose(
            vec_0_3, np.array([2 / 3 / magnitude, 1 / 3 / magnitude, 0.0]), rtol=1e-10
        )

    def test_cell2face_vector(self, mesh2d: Mesh):
        """Test cell-to-face vectors for unstructured mesh"""
        geom = mesh2d.get_geom_assistant()
        cell2face_vects = geom.cell2face_vector

        # Check vectors from cell 0 to its faces
        vec_0_40 = cell2face_vects[0][40].to_numpy()
        magnitude = np.sqrt(2) / 6
        assert np.allclose(
            vec_0_40, np.array([-1 / 6 / magnitude, 1 / 6 / magnitude, 0.0]), rtol=1e-10
        )

        vec_0_0 = cell2face_vects[0][0].to_numpy()
        magnitude = np.sqrt(5) / 6
        assert np.allclose(
            vec_0_0, np.array([-1 / 6 / magnitude, -1 / 3 / magnitude, 0.0]), rtol=1e-10
        )

        vec_0_21 = cell2face_vects[0][21].to_numpy()
        assert np.allclose(
            vec_0_21,
            np.array([1 / 3 / magnitude, 1 / 6 / magnitude, 0.0]),
            rtol=1e-10,
        )

    def test_geom_cache_mechanism(self, mesh2d: Mesh):
        """Test geometry cache mechanism for unstructured mesh"""
        geom = mesh2d.get_geom_assistant()

        # Multiple accesses should return the same result
        face_areas_1 = geom.face_area
        face_areas_2 = geom.face_area
        np.testing.assert_array_equal(face_areas_1, face_areas_2)

        # Test reset method
        geom.reset(mesh2d)
        face_areas_3 = geom.face_area
        np.testing.assert_array_equal(face_areas_1, face_areas_3)
