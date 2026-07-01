# -*- encoding: utf-8 -*-
"""
Tests for MeshPart class.
"""

import numpy as np
import pytest
import torch

from yunmeng.numerics.grids import Grid2D, Grid
from yunmeng.numerics.mesh import GenericMesh, Mesh
from yunmeng.numerics.enums import ElementType

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


@pytest.fixture
def mesh2d_6x6():
    """2D unstructured mesh with 64 triangular cells
    (6x6 grid divided into triangles)."""
    # Create a 6x6 grid of nodes (7x7 points)
    nodes = []
    for j in range(7):
        for i in range(7):
            nodes.append([float(i), float(j), 0.0])

    faces = []
    # Horizontal edges: 7 rows * 6 edges = 42 edges (Indices 0-41)
    for j in range(7):
        for i in range(6):
            faces.append([j * 7 + i, j * 7 + i + 1])

    # Vertical edges: 6 rows * 7 edges = 42 edges (Indices 42-83)
    for j in range(6):
        for i in range(7):
            faces.append([j * 7 + i, (j + 1) * 7 + i])

    cells = []
    for j in range(6):
        for i in range(6):
            # Nodes
            n0 = j * 7 + i  # Bottom-Left
            n1 = j * 7 + i + 1  # Bottom-Right
            n2 = (j + 1) * 7 + i  # Top-Left
            n3 = (j + 1) * 7 + i + 1  # Top-Right

            # Face Indices
            bottom_face = j * 6 + i
            top_face = (j + 1) * 6 + i
            left_face = 42 + j * 7 + i
            right_face = 42 + j * 7 + (i + 1)

            # Diagonal face index
            diagonal_face = len(faces)
            faces.append([n0, n3])

            # Triangle 1: Bottom-Right half (Nodes: n0, n1, n3)
            cells.append([bottom_face, right_face, diagonal_face])

            # Triangle 2: Top-Left half (Nodes: n0, n3, n2)
            cells.append([diagonal_face, top_face, left_face])

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
        total_cells = sum(s.n_core_cells for s in shards)
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
        total_cells = sum(s.n_core_cells for s in shards)
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
            for global_idx in shard.cells[: shard.n_core_cells]:
                assert global_idx in shard.cell_g2l_core
                local_idx = shard.cell_g2l_core[global_idx]
                assert 0 <= local_idx < len(shard.cells)

            # Check that all local faces have a global-to-local mapping
            for global_idx in shard.faces[: shard.n_core_faces]:
                assert global_idx in shard.face_g2l_core
                local_idx = shard.face_g2l_core[global_idx]
                assert 0 <= local_idx < len(shard.faces)

            # Check that all local nodes have a global-to-local mapping
            for global_idx in shard.nodes[: shard.n_core_nodes]:
                assert global_idx in shard.node_g2l_core
                local_idx = shard.node_g2l_core[global_idx]
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
        total_cells = sum(s.n_core_cells for s in shards)
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
        total_cells = sum(s.n_core_cells for s in shards)
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
            for global_idx in shard.cells[: shard.n_core_cells]:
                assert global_idx in shard.cell_g2l_core
                local_idx = shard.cell_g2l_core[global_idx]
                assert 0 <= local_idx < len(shard.cells)

            # Check that all local faces have a global-to-local mapping
            for global_idx in shard.faces[: shard.n_core_faces]:
                assert global_idx in shard.face_g2l_core
                local_idx = shard.face_g2l_core[global_idx]
                assert 0 <= local_idx < len(shard.faces)

            # Check that all local nodes have a global-to-local mapping
            for global_idx in shard.nodes[: shard.n_core_nodes]:
                assert global_idx in shard.node_g2l_core
                local_idx = shard.node_g2l_core[global_idx]
                assert 0 <= local_idx < len(shard.nodes)
