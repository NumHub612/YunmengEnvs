# -*- encoding: utf-8 -*-
"""
Tests for MeshPart class.
"""
import numpy as np
import pytest
import torch

from core.numerics.mesh.grids import Grid2D, Grid
from core.numerics.mesh.meshes import GenericMesh, Mesh
from core.numerics.enums import ElementType

# ============================================
# region Fixtures
# ============================================


@pytest.fixture
def grid2d_4x4():
    "Grid with 16 cells (4x4)"
    x_positions = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    y_positions = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    return Grid2D(x_positions, y_positions)


@pytest.fixture
def grid2d_6x6():
    "Grid with 36 cells (6x6)"
    x_positions = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    y_positions = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    return Grid2D(x_positions, y_positions)


@pytest.fixture
def mesh2d_4x4():
    """2D unstructured mesh with 16 triangular cells
    (4x4 grid divided into triangles)."""
    # Create a 4x4 grid of nodes
    nodes = []
    for j in range(5):
        for i in range(5):
            nodes.append([float(i), float(j), 0.0])

    # Create faces (horizontal and vertical edges)
    faces = []
    # Horizontal edges
    for j in range(5):
        for i in range(4):
            faces.append([j * 5 + i, j * 5 + i + 1])
    # Vertical edges
    for j in range(4):
        for i in range(5):
            faces.append([j * 5 + i, (j + 1) * 5 + i])

    # Create cells (each grid cell divided into 2 triangles)
    cells = []
    face_id = 0
    for j in range(4):
        for i in range(4):
            # Bottom-left node
            n0 = j * 5 + i
            # Bottom-right node
            n1 = j * 5 + i + 1
            # Top-left node
            n2 = (j + 1) * 5 + i
            # Top-right node
            n3 = (j + 1) * 5 + i + 1

            # Bottom edge
            bottom_face = j * 4 + i
            # Left edge
            left_face = 20 + j * 5 + i
            # Right edge
            right_face = 20 + j * 5 + i + 1
            # Top edge
            top_face = (j + 1) * 4 + i

            # Diagonal edge (will be added to faces)
            diagonal_face = len(faces)
            faces.append([n0, n3])

            # Two triangles per cell
            # Triangle 1: bottom, left, diagonal
            cells.append([bottom_face, left_face, diagonal_face])
            # Triangle 2: top, right, diagonal
            cells.append([top_face, right_face, diagonal_face])

    return GenericMesh(nodes, faces, cells)


@pytest.fixture
def mesh2d_6x6():
    """2D unstructured mesh with 36 triangular cells
    (6x6 grid divided into triangles)."""
    # Create a 6x6 grid of nodes
    nodes = []
    for j in range(7):
        for i in range(7):
            nodes.append([float(i), float(j), 0.0])

    # Create faces (horizontal and vertical edges)
    faces = []
    # Horizontal edges
    for j in range(7):
        for i in range(6):
            faces.append([j * 7 + i, j * 7 + i + 1])
    # Vertical edges
    for j in range(6):
        for i in range(7):
            faces.append([j * 7 + i, (j + 1) * 7 + i])

    # Create cells (each grid cell divided into 2 triangles)
    cells = []
    for j in range(6):
        for i in range(6):
            # Bottom-left node
            n0 = j * 7 + i
            # Bottom-right node
            n1 = j * 7 + i + 1
            # Top-left node
            n2 = (j + 1) * 7 + i
            # Top-right node
            n3 = (j + 1) * 7 + i + 1

            # Bottom edge
            bottom_face = j * 6 + i
            # Left edge
            left_face = 42 + j * 7 + i
            # Right edge
            right_face = 42 + j * 7 + i + 1
            # Top edge
            top_face = (j + 1) * 6 + i

            # Diagonal edge (will be added to faces)
            diagonal_face = len(faces)
            faces.append([n0, n3])

            # Two triangles per cell
            # Triangle 1: bottom, left, diagonal
            cells.append([bottom_face, left_face, diagonal_face])
            # Triangle 2: top, right, diagonal
            cells.append([top_face, right_face, diagonal_face])

    return GenericMesh(nodes, faces, cells)


# ============================================
# region Structured mesh
# ============================================


class TestStructuredGrid:
    """Test cases for MeshPart class with structured grids."""

    def test_partition_single_shard_cpu(self, grid2d_4x4: Grid):
        """Test partitioning a structured grid into a single shard on CPU."""
        part = grid2d_4x4.get_part_assistant()
        shards = part.partition(num_shards=1, device="cpu")

        assert len(shards) == 1
        assert shards[0].shard_id == 0
        assert shards[0].gpu.type == "cpu"
        assert len(shards[0].cells) == grid2d_4x4.cell_count
        assert len(shards[0].faces) == grid2d_4x4.face_count
        assert len(shards[0].nodes) == grid2d_4x4.node_count

    def test_partition_multiple_shards_cpu(self, grid2d_6x6: Grid):
        """Test partitioning a structured grid into multiple shards on CPU."""
        part = grid2d_6x6.get_part_assistant()
        shards = part.partition(num_shards=4, device="cpu")

        assert len(shards) == 4
        for i, shard in enumerate(shards):
            assert shard.shard_id == i
            assert shard.gpu.type == "cpu"
            # Check that all cells are distributed across shards
            assert len(shard.cells) > 0

        # Check that all cells are accounted for
        total_cells = sum(len(shard.cell_g2l) for shard in shards)
        assert total_cells == part.get_size(ElementType.CELL)

    def test_partition_single_shard_gpu(self, grid2d_4x4: Grid):
        """Test partitioning a structured grid into a single shard on GPU."""
        if not torch.cuda.is_available():
            pytest.skip("Skip GPU test if CUDA is not available")

        part = grid2d_4x4.get_part_assistant()
        shards = part.partition(num_shards=1, device="cuda", gpus=[0])

        assert len(shards) == 1
        assert shards[0].shard_id == 0
        assert shards[0].gpu.type == "cuda"
        assert shards[0].gpu.index == 0
        assert len(shards[0].cells) == grid2d_4x4.cell_count

    def test_partition_multiple_shards_gpu(self, grid2d_6x6: Grid):
        """Test partitioning a structured grid into multiple shards on GPU."""
        if not torch.cuda.is_available():
            pytest.skip("Skip GPU test if CUDA is not available")

        part = grid2d_6x6.get_part_assistant()
        shards = part.partition(num_shards=2, device="cuda", gpus=[0, 1])

        assert len(shards) == 2
        for i, shard in enumerate(shards):
            assert shard.shard_id == i
            assert shard.gpu.type == "cuda"
            assert shard.gpu.index == i
            assert len(shard.cells) > 0

        # Check that all cells are accounted for
        total_cells = sum(len(shard.cell_g2l) for shard in shards)
        assert total_cells == part.get_size(ElementType.CELL)

    def test_cell_parts_property(self, grid2d_6x6: Grid):
        """Test cell_parts property."""
        part = grid2d_6x6.get_part_assistant()
        cell_parts = part.cell_parts
        shards = part.num_shards

        assert cell_parts is not None
        assert len(cell_parts) == grid2d_6x6.cell_count
        # Check that all cells are assigned to a partition
        assert all(0 <= p < shards for p in cell_parts)

    def test_shards_property(self, grid2d_6x6: Grid):
        """Test shards property."""
        part = grid2d_6x6.get_part_assistant()
        shards = part.shards

        assert shards is not None
        for i, shard in enumerate(shards):
            assert shard.shard_id == i

    def test_halo_information(self, grid2d_6x6: Grid):
        """Test halo information in multi-shard partitioning."""
        part = grid2d_6x6.get_part_assistant()
        shards = part.partition(num_shards=4, device="cpu")

        # For multi-shard partitions, some shards should have neighbors
        if len(shards) > 1:
            # At least some shards should have neighbors
            has_neighbors = any(len(s.cell_halo.neighbours) > 0 for s in shards)
            assert has_neighbors

    def test_mapping_consistency(self, grid2d_6x6: Grid):
        """Test consistency of global-to-local mappings."""
        part = grid2d_6x6.get_part_assistant()
        shards = part.partition(num_shards=4, device="cpu")

        for shard in shards:
            # Check that all local cells have a global-to-local mapping
            for global_idx in shard.cells:
                if global_idx not in shard.halo_g2l:  # Skip ghost cells
                    assert global_idx in shard.cell_g2l
                    local_idx = shard.cell_g2l[global_idx]
                    assert 0 <= local_idx < len(shard.cells)

            # Check that all local faces have a global-to-local mapping
            for global_idx in shard.faces:
                assert global_idx in shard.face_g2l
                local_idx = shard.face_g2l[global_idx]
                assert 0 <= local_idx < len(shard.faces)

            # Check that all local nodes have a global-to-local mapping
            for global_idx in shard.nodes:
                assert global_idx in shard.node_g2l
                local_idx = shard.node_g2l[global_idx]
                assert 0 <= local_idx < len(shard.nodes)


# ============================================
# region un-Structured mesh
# ============================================


class TestUnstructuredMesh:
    """Test cases for MeshPart class with unstructured meshes."""

    def test_partition_single_shard_cpu(self, mesh2d_4x4: Mesh):
        """Test partitioning an unstructured mesh into a single shard on CPU."""
        part = mesh2d_4x4.get_part_assistant()
        shards = part.partition(num_shards=1, device="cpu")

        assert len(shards) == 1
        assert shards[0].shard_id == 0
        assert shards[0].gpu.type == "cpu"
        assert len(shards[0].cells) == mesh2d_4x4.cell_count
        assert len(shards[0].faces) == mesh2d_4x4.face_count
        assert len(shards[0].nodes) == mesh2d_4x4.node_count

    def test_partition_multiple_shards_cpu(self, mesh2d_6x6: Mesh):
        """Test partitioning an unstructured mesh into multiple shards on CPU."""
        part = mesh2d_6x6.get_part_assistant()
        shards = part.partition(num_shards=4, device="cpu")

        assert len(shards) == 4
        for i, shard in enumerate(shards):
            assert shard.shard_id == i
            assert shard.gpu.type == "cpu"
            # Check that all cells are distributed across shards
            assert len(shard.cells) > 0

        # Check that all cells are accounted for
        total_cells = sum(len(shard.cell_g2l) for shard in shards)
        assert total_cells == part.get_size(ElementType.CELL)

    def test_partition_single_shard_gpu(self, mesh2d_4x4: Mesh):
        """Test partitioning an unstructured mesh into a single shard on GPU."""
        if not torch.cuda.is_available():
            pytest.skip("Skip GPU test if CUDA is not available")

        part = mesh2d_4x4.get_part_assistant()
        shards = part.partition(num_shards=1, device="cuda", gpus=[0])

        assert len(shards) == 1
        assert shards[0].shard_id == 0
        assert shards[0].gpu.type == "cuda"
        assert shards[0].gpu.index == 0
        assert len(shards[0].cells) == mesh2d_4x4.cell_count

    def test_partition_multiple_shards_gpu(self, mesh2d_6x6: Mesh):
        """Test partitioning an unstructured mesh into multiple shards on GPU."""
        if not torch.cuda.is_available():
            pytest.skip("Skip GPU test if CUDA is not available")

        part = mesh2d_6x6.get_part_assistant()
        shards = part.partition(num_shards=2, device="cuda", gpus=[0, 1])

        assert len(shards) == 2
        for i, shard in enumerate(shards):
            assert shard.shard_id == i
            assert shard.gpu.type == "cuda"
            assert shard.gpu.index == i
            assert len(shard.cells) > 0

        # Check that all cells are accounted for
        total_cells = sum(len(shard.cell_g2l) for shard in shards)
        assert total_cells == part.get_size(ElementType.CELL)

    def test_cell_parts_property(self, mesh2d_6x6: Mesh):
        """Test cell_parts property."""
        part = mesh2d_6x6.get_part_assistant()
        cell_parts = part.cell_parts
        shards = part.num_shards

        assert cell_parts is not None
        assert len(cell_parts) == mesh2d_6x6.cell_count
        # Check that all cells are assigned to a partition
        assert all(0 <= p < shards for p in cell_parts)

    def test_shards_property(self, mesh2d_6x6: Mesh):
        """Test shards property."""
        part = mesh2d_6x6.get_part_assistant()
        shards = part.shards

        assert shards is not None
        for i, shard in enumerate(shards):
            assert shard.shard_id == i

    def test_halo_information(self, mesh2d_6x6: Mesh):
        """Test halo information in multi-shard partitioning."""
        part = mesh2d_6x6.get_part_assistant()
        shards = part.partition(num_shards=4, device="cpu")

        # For multi-shard partitions, some shards should have neighbors
        if len(shards) > 1:
            # At least some shards should have neighbors
            has_neighbors = any(len(s.cell_halo.neighbours) > 0 for s in shards)
            assert has_neighbors

    def test_mapping_consistency(self, mesh2d_6x6: Mesh):
        """Test consistency of global-to-local mappings."""
        part = mesh2d_6x6.get_part_assistant()
        shards = part.partition(num_shards=4, device="cpu")

        for shard in shards:
            # Check that all local cells have a global-to-local mapping
            for global_idx in shard.cells:
                if global_idx not in shard.halo_g2l:  # Skip ghost cells
                    assert global_idx in shard.cell_g2l
                    local_idx = shard.cell_g2l[global_idx]
                    assert 0 <= local_idx < len(shard.cells)

            # Check that all local faces have a global-to-local mapping
            for global_idx in shard.faces:
                assert global_idx in shard.face_g2l
                local_idx = shard.face_g2l[global_idx]
                assert 0 <= local_idx < len(shard.faces)

            # Check that all local nodes have a global-to-local mapping
            for global_idx in shard.nodes:
                assert global_idx in shard.node_g2l
                local_idx = shard.node_g2l[global_idx]
                assert 0 <= local_idx < len(shard.nodes)
