# -*- encoding: utf-8 -*-
"""
Unit tests for MeshGeom.
"""

import pytest
import numpy as np
from core.numerics.mesh.grids import Grid2D, Grid
from core.numerics.mesh.meshes import GenericMesh, Mesh
from core.numerics.mesh.elements import Coordinate
from core.numerics.algos.geoms import calculate_distance


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


class TestGrid2DGeom:
    """Test Grid2D mesh geometry"""

    def test_face_perimeter_4x4(self, grid2d: Grid):
        """Test face perimeter for 4x4 Grid2D mesh"""
        geom = grid2d.get_geom_assistant()
        face_perimeters = geom.face_perimeter

        # In a 4x4 grid with uniform spacing of 1.0,
        # each face should have perimeter 1.0
        assert len(face_perimeters) == 24
        np.testing.assert_allclose(face_perimeters, np.ones(24), rtol=1e-10)

    def test_face_area_4x4(self, grid2d: Grid):
        """Test face area for 4x4 Grid2D mesh"""
        geom = grid2d.get_geom_assistant()
        face_areas = geom.face_area

        # In 2D, face area equals face perimeter
        face_perimeters = geom.face_perimeter
        np.testing.assert_allclose(face_areas, face_perimeters, rtol=1e-10)

    def test_face_normal_4x4(self, grid2d: Grid):
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

    def test_cell_volume_4x4(self, grid2d: Grid):
        """Test cell volume (area in 2D) for 4x4 Grid2D mesh"""
        geom = grid2d.get_geom_assistant()
        cell_volumes = geom.cell_volume

        # In a 4x4 grid with uniform spacing of 1.0,
        # each cell should have volume(area) of 1.0
        assert len(cell_volumes) == 9
        np.testing.assert_allclose(cell_volumes, np.ones(9), rtol=1e-10)

    def test_cell_surface_4x4(self, grid2d: Grid):
        """Test cell surface (perimeter in 2D) for 4x4 Grid2D mesh"""
        geom = grid2d.get_geom_assistant()
        cell_surfaces = geom.cell_surface

        # Each cell should have surface (perimeter) of 4.0
        assert len(cell_surfaces) == 9
        np.testing.assert_allclose(cell_surfaces, 4.0 * np.ones(9), rtol=1e-10)

    def test_cell2cell_distance_4x4(self, grid2d: Grid):
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

    def test_cell2face_distance_4x4(self, grid2d: Grid):
        """Test cell-to-face distances for 4x4 Grid2D mesh"""
        geom = grid2d.get_geom_assistant()
        cell2face_dists = geom.cell2face_distance

        # Check distances between cells and their faces
        # For a uniform grid, each cell should be 0.5 away from each of its faces
        for cell_id in range(grid2d.cell_count):
            for face_id, dist in cell2face_dists[cell_id].items():
                assert abs(dist - 0.5) < 1e-10

    def test_cell2cell_vector_4x4(self, grid2d: Grid):
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

    def test_cell2face_vector_4x4(self, grid2d: Grid):
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
    """Test GenericMesh geometry"""

    def test_face_perimeter_4x4(self, mesh2d: Mesh):
        """Test face perimeter for 4x4 GenericMesh"""
        geom = mesh2d.get_geom_assistant()
        face_perimeters = geom.face_perimeter

        # In a 4x4 grid with uniform spacing of 1.0,
        # each face should have perimeter 1.0
        assert len(face_perimeters) == 24
        np.testing.assert_allclose(face_perimeters, np.ones(24), rtol=1e-10)

    def test_face_area_4x4(self, mesh2d: Mesh):
        """Test face area for 4x4 GenericMesh"""
        geom = mesh2d.get_geom_assistant()
        face_areas = geom.face_area

        # In 2D, face area equals face perimeter
        face_perimeters = geom.face_perimeter
        np.testing.assert_allclose(face_areas, face_perimeters, rtol=1e-10)

    def test_face_normal_4x4(self, mesh2d: Mesh):
        """Test face normal vectors for 4x4 GenericMesh"""
        geom = mesh2d.get_geom_assistant()
        face_normals = geom.face_normal

        # Each normal should be a unit vector
        for normal in face_normals:
            norm = np.linalg.norm(normal.to_numpy())
            assert abs(norm - 1.0) < 1e-10

        # Horizontal faces should have normals in y-direction
        # Faces 0-2, 3-5, 6-8, 9-11 are horizontal
        for i in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]:
            normal = face_normals[i].to_numpy()
            assert abs(normal[0]) < 1e-10  # x-component should be ~0
            assert abs(normal[2]) < 1e-10  # z-component should be ~0

        # Vertical faces should have normals in x-direction
        # Faces 12-14, 15-17, 18-20, 21-23 are vertical
        for i in [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]:
            normal = face_normals[i].to_numpy()
            assert abs(normal[1]) < 1e-10  # y-component should be ~0
            assert abs(normal[2]) < 1e-10  # z-component should be ~0

    def test_cell_volume_4x4(self, mesh2d: Mesh):
        """Test cell volume (area in 2D) for 4x4 GenericMesh"""
        geom = mesh2d.get_geom_assistant()
        cell_volumes = geom.cell_volume

        # In a 4x4 grid with uniform spacing of 1.0,
        # each cell should have volume(area) of 1.0
        assert len(cell_volumes) == 9
        np.testing.assert_allclose(cell_volumes, np.ones(9), rtol=1e-10)

    def test_cell_surface_4x4(self, mesh2d: Mesh):
        """Test cell surface (perimeter in 2D) for 4x4 GenericMesh"""
        geom = mesh2d.get_geom_assistant()
        cell_surfaces = geom.cell_surface

        # Each cell should have surface (perimeter) of 4.0
        assert len(cell_surfaces) == 9
        np.testing.assert_allclose(cell_surfaces, 4.0 * np.ones(9), rtol=1e-10)

    def test_cell2cell_distance_4x4(self, mesh2d: Mesh):
        """Test cell-to-cell distances for 4x4 GenericMesh"""
        geom = mesh2d.get_geom_assistant()
        cell2cell_dists = geom.cell2cell_distance

        # Check distances between neighboring cells
        # Cell 0 and cell 1 should be 1.0 apart
        assert abs(cell2cell_dists[0][1] - 1.0) < 1e-10
        # Cell 0 and cell 3 should be 1.0 apart
        assert abs(cell2cell_dists[0][3] - 1.0) < 1e-10
        # Cell 4 and cell 7 should be 1.0 apart
        assert abs(cell2cell_dists[4][7] - 1.0) < 1e-10

    def test_cell2face_distance_4x4(self, mesh2d: Mesh):
        """Test cell-to-face distances for 4x4 GenericMesh"""
        geom = mesh2d.get_geom_assistant()
        cell2face_dists = geom.cell2face_distance

        # Check distances between cells and their faces
        # For a uniform grid, each cell should be 0.5 away from each of its faces
        for cell_id in range(mesh2d.cell_count):
            for face_id, dist in cell2face_dists[cell_id].items():
                assert abs(dist - 0.5) < 1e-10

    def test_cell2cell_vector_4x4(self, mesh2d: Mesh):
        """Test cell-to-cell vectors for 4x4 GenericMesh"""
        geom = mesh2d.get_geom_assistant()
        cell2cell_vects = geom.cell2cell_vector

        # Check vectors between neighboring cells
        # Cell 0 to cell 1 should be in positive x-direction
        vec_0_1 = cell2cell_vects[0][1].to_numpy()
        assert abs(vec_0_1[0] - 1.0) < 1e-10
        assert abs(vec_0_1[1]) < 1e-10
        assert abs(vec_0_1[2]) < 1e-10

        # Cell 0 to cell 3 should be in positive y-direction
        vec_0_3 = cell2cell_vects[0][3].to_numpy()
        assert abs(vec_0_3[0]) < 1e-10
        assert abs(vec_0_3[1] - 1.0) < 1e-10
        assert abs(vec_0_3[2]) < 1e-10

    def test_cell2face_vector_4x4(self, mesh2d: Mesh):
        """Test cell-to-face vectors for 4x4 GenericMesh"""
        geom = mesh2d.get_geom_assistant()
        cell2face_vects = geom.cell2face_vector

        # Check vectors from cell 0 to its faces
        # Cell 0 to face 0 should be in negative y-direction
        vec_0_0 = cell2face_vects[0][0].to_numpy()
        assert abs(vec_0_0[0]) < 1e-10
        assert abs(vec_0_0[1] + 1.0) < 1e-10
        assert abs(vec_0_0[2]) < 1e-10

        # Cell 0 to face 0 should be in negative x-direction
        vec_0_12 = cell2face_vects[0][12].to_numpy()
        assert abs(vec_0_12[0] + 1.0) < 1e-10
        assert abs(vec_0_12[1]) < 1e-10
        assert abs(vec_0_12[2]) < 1e-10

    def test_geom_cache_mechanism(self, mesh2d: Mesh):
        """Test geometry cache mechanism"""
        geom = mesh2d.get_geom_assistant()

        # Multiple accesses should return the same result
        face_areas_1 = geom.face_area
        face_areas_2 = geom.face_area
        np.testing.assert_array_equal(face_areas_1, face_areas_2)

        # Test reset method
        geom.reset(mesh2d)
        face_areas_3 = geom.face_area
        np.testing.assert_array_equal(face_areas_1, face_areas_3)

    def test_calculate_distance(self):
        """Test calculate_distance function"""
        coord1 = Coordinate(0.0, 0.0, 0.0)
        coord2 = Coordinate(1.0, 0.0, 0.0)
        coord3 = Coordinate(0.0, 1.0, 0.0)
        coord4 = Coordinate(0.0, 0.0, 1.0)

        # Test distance calculations
        assert abs(calculate_distance(coord1, coord2) - 1.0) < 1e-10
        assert abs(calculate_distance(coord1, coord3) - 1.0) < 1e-10
        assert abs(calculate_distance(coord1, coord4) - 1.0) < 1e-10
        assert abs(calculate_distance(coord1, coord1) - 0.0) < 1e-10

        # Test distance between coordinates at (0,0,0) and (1,1,1)
        coord5 = Coordinate(1.0, 1.0, 1.0)
        expected_dist = np.sqrt(3.0)
        assert abs(calculate_distance(coord1, coord5) - expected_dist) < 1e-10
