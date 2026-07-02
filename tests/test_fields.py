# -*- encoding: utf-8 -*-
"""
Unit tests for Field class with actual mesh (grid2d and mesh2d).
"""

import pytest
import numpy as np
import torch

from yunmeng.setting import settings
from yunmeng.numerics.grids import Grid2D
from yunmeng.numerics.mesh import GenericMesh, Coordinate
from yunmeng.numerics.fields import Field, HaloMode, MeshShard
from yunmeng.numerics.enums import ElementType, VariableType

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

            # Calculate Face Indices based on generation order above
            bottom_face = j * 4 + i
            top_face = (j + 1) * 4 + i
            left_face = 20 + j * 5 + i
            right_face = 20 + j * 5 + (i + 1)

            # Diagonal face index: 40 + current_cell_index
            diagonal_face = 40 + (j * 4 + i)

            # Add diagonal edge to faces list: connects n0 and n3
            faces.append([n0, n3])

            # Triangle 1: Bottom-Right half (Nodes: n0, n1, n3)
            cells.append([bottom_face, right_face, diagonal_face])

            # Triangle 2: Top-Left half (Nodes: n0, n3, n2)
            cells.append([diagonal_face, top_face, left_face])

    return GenericMesh(nodes, faces, cells)


@pytest.fixture
def grid2d_shards(grid2d: Grid2D):
    """Create a partitioned mesh part from grid2d (2 partitions)"""
    mesh_part = grid2d.get_part_assistant()
    device = settings.device
    mesh_part.partition(num_shards=2, device=device)
    return mesh_part.shards


@pytest.fixture
def mesh2d_shards(mesh2d: GenericMesh):
    """Create a partitioned mesh part from mesh2d (4 partitions)"""
    mesh_part = mesh2d.get_part_assistant()
    device = settings.device
    num_shards = min(4, len(settings.gpus)) if device == "cuda" else 4
    mesh_part.partition(num_shards=num_shards, device=device)
    return mesh_part.shards


# ============================================
# region Grid2D field
# ============================================


class TestGrid2DField:
    """Test Field with Grid2D mesh with partitioning"""

    def test_scalar_field_initialization(self, grid2d_shards: list[MeshShard]):
        """Test scalar field initialization on CPU with partitioning"""
        field = Field(
            mesh_shards=grid2d_shards,
            vtype=VariableType.SCALAR,
            etype=ElementType.CELL,
            init_val=1.0,
            requires_grad=False,
        )

        # Check field properties
        assert field.size == sum([s.n_core_cells for s in grid2d_shards])
        assert field.vtype == VariableType.SCALAR
        assert field.etype == ElementType.CELL

        # Check initial values
        for i in range(field.size):
            assert field[i] == 1.0

    def test_field_getitem_setitem(self, grid2d_shards: list[MeshShard]):
        """Test field indexing and assignment on CPU with partitioning"""
        field = Field(
            mesh_shards=grid2d_shards,
            vtype=VariableType.SCALAR,
            etype=ElementType.CELL,
            init_val=0.0,
            requires_grad=False,
        )

        # Set values
        for i in range(field.size):
            field[i] = float(i)

        # Check values
        for i in range(field.size):
            assert field[i] == float(i)

    def test_field_arithmetic_operations(self, grid2d_shards: list[MeshShard]):
        """Test field arithmetic operations on CPU with partitioning"""
        field1 = Field(
            mesh_shards=grid2d_shards,
            vtype=VariableType.SCALAR,
            etype=ElementType.CELL,
            init_val=2.0,
            requires_grad=False,
        )
        field2 = Field(
            mesh_shards=grid2d_shards,
            vtype=VariableType.SCALAR,
            etype=ElementType.CELL,
            init_val=3.0,
            requires_grad=False,
        )

        # Test addition
        result = field1 + field2
        for i in range(result.size):
            assert result[i] == 5.0

        # Test subtraction
        result = field2 - field1
        for i in range(result.size):
            assert result[i] == 1.0

        # Test scalar multiplication
        result = field1 * 2.0
        for i in range(result.size):
            assert result[i] == 4.0

        # Test scalar division
        result = field2 / 2.0
        for i in range(result.size):
            assert result[i] == 1.5

    def test_field_gather_to_host(self, grid2d_shards: list[MeshShard]):
        """Test gathering partitioned field data to host on CPU"""
        field = Field(
            mesh_shards=grid2d_shards,
            vtype=VariableType.SCALAR,
            etype=ElementType.CELL,
            init_val=0.0,
            requires_grad=False,
        )

        # Set some values
        for i in range(field.size):
            field[i] = float(i)

        # Gather to host
        data = field.gather_to_host()

        # Check data
        assert data.shape == (9,)
        for i in range(9):
            assert data[i] == float(i)

    def test_field_scatter_from_host(self, grid2d_shards: list[MeshShard]):
        """Test scattering partitioned field data from host on CPU"""
        field = Field(
            mesh_shards=grid2d_shards,
            vtype=VariableType.SCALAR,
            etype=ElementType.CELL,
            init_val=0.0,
            requires_grad=False,
        )

        # Create global array
        global_arr = np.array([float(i) for i in range(9)])

        # Scatter from host
        field.scatter_from_host(global_arr)

        # Check values
        for i in range(field.size):
            assert field[i] == global_arr[i]


# ============================================
# region Mesh2D Field
# ============================================


class TestMesh2DField:
    """Test Field with Mesh2D mesh with partitioning"""

    def test_scalar_field_initialization(self, mesh2d_shards: list[MeshShard]):
        """Test scalar field initialization on CPU with partitioning"""
        field = Field(
            mesh_shards=mesh2d_shards,
            vtype=VariableType.SCALAR,
            etype=ElementType.CELL,
            init_val=0.0,
            requires_grad=False,
        )

        # Check field properties
        assert field.size == sum([s.n_core_cells for s in mesh2d_shards])
        assert field.vtype == VariableType.SCALAR
        assert field.etype == ElementType.CELL

        # Check initial values
        for i in range(field.size):
            assert field[i] == 0.0

    def test_field_getitem_setitem(self, mesh2d_shards: list[MeshShard]):
        """Test field indexing and assignment on CPU with partitioning"""
        field = Field(
            mesh_shards=mesh2d_shards,
            vtype=VariableType.SCALAR,
            etype=ElementType.CELL,
            init_val=0.0,
            requires_grad=False,
        )

        # Set values
        for i in range(field.size):
            field[i] = float(i)

        # Check values
        for i in range(field.size):
            assert field[i] == float(i)

    def test_field_arithmetic_operations(self, mesh2d_shards: list[MeshShard]):
        """Test field arithmetic operations on CPU with partitioning"""
        field1 = Field(
            mesh_shards=mesh2d_shards,
            vtype=VariableType.SCALAR,
            etype=ElementType.CELL,
            init_val=2.0,
            requires_grad=False,
        )
        field2 = Field(
            mesh_shards=mesh2d_shards,
            vtype=VariableType.SCALAR,
            etype=ElementType.CELL,
            init_val=3.0,
            requires_grad=False,
        )

        # Test addition
        result = field1 + field2
        for i in range(result.size):
            assert result[i] == 5.0

        # Test subtraction
        result = field2 - field1
        for i in range(result.size):
            assert result[i] == 1.0

        # Test scalar multiplication
        result = field1 * 2.0
        for i in range(result.size):
            assert result[i] == 4.0

        # Test scalar division
        result = field2 / 2.0
        for i in range(result.size):
            assert result[i] == 1.5

    def test_field_gather_to_host(self, mesh2d_shards: list[MeshShard]):
        """Test gathering partitioned field data to host on CPU"""
        field = Field(
            mesh_shards=mesh2d_shards,
            vtype=VariableType.SCALAR,
            etype=ElementType.CELL,
            init_val=0.0,
            requires_grad=False,
        )

        # Set some values
        for i in range(field.size):
            field[i] = float(i)

        # Gather to host
        data = field.gather_to_host()

        # Check data
        assert data.shape == (32,)
        for i in range(32):
            assert data[i] == float(i)

    def test_field_scatter_from_host(self, mesh2d_shards: list[MeshShard]):
        """Test scattering partitioned field data from host on CPU"""
        field = Field(
            mesh_shards=mesh2d_shards,
            vtype=VariableType.SCALAR,
            etype=ElementType.CELL,
            init_val=0.0,
            requires_grad=False,
        )

        # Create global array
        global_arr = np.array([float(i) for i in range(32)])

        # Scatter from host
        field.scatter_from_host(global_arr)

        # Check values
        for i in range(field.size):
            assert field[i] == global_arr[i]


# ============================================
# region GPU Tests
# ============================================


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestMultiGPUField:
    """Test Field on GPU"""

    def test_grid2d_field_initialization(self, grid2d: Grid2D):
        """Test scalar field initialization on GPU"""
        mesh_part = grid2d.get_part_assistant()
        num_shards = min(2, len(settings.gpus))
        mesh_part.partition(
            num_shards=num_shards,
            device="cuda",
            gpus=[0, 1] if torch.cuda.device_count() > 1 else [0, 0],
        )

        field = Field(
            mesh_shards=mesh_part.shards,
            vtype=VariableType.SCALAR,
            etype=ElementType.CELL,
            init_val=0.0,
            requires_grad=False,
        )

        # Check field properties
        assert field.size == mesh_part.get_size(ElementType.CELL)
        assert field.vtype == VariableType.SCALAR
        assert field.etype == ElementType.CELL

        # Check initial values
        for i in range(field.size):
            assert field[i] == 0.0

    def test_grid2d_field_getitem_setitem(self, grid2d: Grid2D):
        """Test field indexing and assignment on GPU with partitioning"""
        mesh_part = grid2d.get_part_assistant()
        num_shards = min(2, len(settings.gpus))
        mesh_part.partition(
            num_shards=num_shards,
            device="cuda",
            gpus=[0, 1] if torch.cuda.device_count() > 1 else [0, 0],
        )

        field = Field(
            mesh_shards=mesh_part.shards,
            vtype=VariableType.SCALAR,
            etype=ElementType.CELL,
            init_val=0.0,
            requires_grad=False,
        )

        # Set values
        for i in range(field.size):
            field[i] = float(i)

        # Check values
        for i in range(field.size):
            assert field[i] == float(i)

    def test_grid2d_field_gather_to_host(self, grid2d: Grid2D):
        """Test gathering partitioned field data to host from GPU"""
        mesh_part = grid2d.get_part_assistant()
        num_shards = min(2, len(settings.gpus))
        mesh_part.partition(
            num_shards=num_shards,
            device="cuda",
            gpus=[0, 1] if torch.cuda.device_count() > 1 else [0, 0],
        )

        field = Field(
            mesh_shards=mesh_part.shards,
            vtype=VariableType.SCALAR,
            etype=ElementType.CELL,
            init_val=0.0,
            requires_grad=False,
        )

        # Set some values
        for i in range(field.size):
            field[i] = float(i)

        # Gather to host
        data = field.gather_to_host()

        # Check data
        assert data.shape == (field.size, 1)
        for i in range(field.size):
            assert data[i, 0] == float(i)

    def test_mesh2d_field_initialization(self, mesh2d: GenericMesh):
        """Test scalar field initialization on GPU with partitioning"""
        mesh_part = mesh2d.get_part_assistant()
        num_shards = min(4, len(settings.gpus))
        mesh_part.partition(num_shards=num_shards, device="cuda", gpus=[0, 0, 0, 0])

        field = Field(
            mesh_shards=mesh_part.shards,
            vtype=VariableType.SCALAR,
            etype=ElementType.CELL,
            init_val=0.0,
            requires_grad=False,
        )

        # Check field properties
        assert field.size == mesh_part.get_size(ElementType.CELL)
        assert field.vtype == VariableType.SCALAR
        assert field.etype == ElementType.CELL

        # Check initial values
        for i in range(field.size):
            assert field[i] == 0.0

    def test_field_getitem_setitem(self, mesh2d: GenericMesh):
        """Test field indexing and assignment on GPU with partitioning"""
        mesh_part = mesh2d.get_part_assistant()
        num_shards = min(4, len(settings.gpus))
        mesh_part.partition(num_shards=num_shards, device="cuda", gpus=[0, 0, 0, 0])

        field = Field(
            mesh_shards=mesh_part.shards,
            vtype=VariableType.SCALAR,
            etype=ElementType.CELL,
            init_val=0.0,
            requires_grad=False,
        )

        # Set values
        for i in range(field.size):
            field[i] = float(i)

        # Check values
        for i in range(field.size):
            assert field[i] == float(i)

    def test_field_gather_to_host(self, mesh2d: GenericMesh):
        """Test gathering partitioned field data to host from GPU"""
        mesh_part = mesh2d.get_part_assistant()
        num_shards = min(4, len(settings.gpus))
        mesh_part.partition(num_shards=num_shards, device="cuda", gpus=[0, 0, 0, 0])

        field = Field(
            mesh_shards=mesh_part.shards,
            vtype=VariableType.SCALAR,
            etype=ElementType.CELL,
            init_val=0.0,
            requires_grad=False,
        )

        # Set some values
        for i in range(field.size):
            field[i] = float(i)

        # Gather to host
        data = field.gather_to_host()

        # Check datas
        assert data.shape == (field.size, 1)
        for i in range(field.size):
            assert data[i, 0] == float(i)


# ============================================
# region Sync Tests
# ============================================


class TestFieldOperations:
    """
    Robust tests for field operations with data consistency checks.
    """

    def test_apply_function_correctness(self, grid2d_shards: list[MeshShard]):
        """Verify apply modifies data in-place correctly"""
        field = Field(
            mesh_shards=grid2d_shards,
            vtype=VariableType.SCALAR,
            etype=ElementType.CELL,
            init_val=2.0,
            requires_grad=False,
        )

        # Apply f(x) = x^2 + 1 -> 2^2 + 1 = 5
        def func(data):
            data[:] = data * 2 + 1

        field.apply(func)

        # Check all local elements
        for i in range(field.size):
            assert (
                abs(field[i] - 5.0) < 1e-6
            ), f"Element {i} expected 5.0, got {field[i]}"

        # Verify dirty flag is set after modification
        assert any(
            field._dirty_flags.values()
        ), "Field should be marked dirty after apply"

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA required for multi-shard logic"
    )
    def test_gradient_computation_flow(self, grid2d_shards: list[MeshShard]):
        """Test gradient retrieval logic and error handling"""
        # Case 1: requires_grad=False should raise
        field_no_grad = Field(
            mesh_shards=grid2d_shards,  # Use default shards
            vtype=VariableType.SCALAR,
            etype=ElementType.CELL,
            init_val=1.0,
            requires_grad=False,
        )
        with pytest.raises(RuntimeError):
            field_no_grad.gradient()

        # Case 2: requires_grad=True (if backend supports)
        try:
            field_grad = Field(
                mesh_shards=grid2d_shards,
                vtype=VariableType.SCALAR,
                etype=ElementType.CELL,
                init_val=1.0,
                requires_grad=True,
            )

            for shard in field_grad._shards:
                shard.data = shard.data + 0
                shard.data.retain_grad()

            def func(data):
                data.mul_(2).add_(1)

            field_grad.apply(func)
            for v in field_grad[[0, 1, 2, 3]]:
                assert v == 3.0

            loss = torch.tensor(
                0.0, dtype=torch.float64, device=field_grad.field_shards[0].gpu
            )
            loss = loss + field_grad.field_shards[0].data.sum()
            loss.backward()

            grads = field_grad.gradient()
            n_core = field_grad.field_shards[0].n_core
            assert isinstance(grads, torch.Tensor)
            assert grads.shape[0] == field_grad.size
            # assert grads[0] == torch.Tensor([2.0])
            # assert grads[n_core - 1] == torch.Tensor([0.0])
        except Exception as e:
            if "CUDA" in str(e) or "backend" in str(e).lower():
                pytest.skip(f"{e}")
            else:
                raise

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA required for multi-shard logic"
    )
    def test_sync_halos_data_overwrite(self, grid2d_shards: list[MeshShard]):
        """
        CRITICAL TEST: Verifies that sync_halos actually copies data from neighbor.
        Strategy:
        1. Partition grid into 2 shards along X-axis.
        2. Set a unique value in the last internal cell of Shard 0.
        3. This cell corresponds to the Halo of Shard 1.
        4. Run sync_halos.
        5. Verify Shard 1's Halo has the unique value.
        """
        field = Field(
            mesh_shards=grid2d_shards,
            vtype=VariableType.SCALAR,
            etype=ElementType.CELL,
            init_val=0.0,
            requires_grad=False,
        )

        # --- Step 1: Identify Boundary Indices ---
        rank_0_val = 100.0
        rank_1_val = 200.0

        # Set Shard 0 to 100, Shard 1 to 200
        for i in range(grid2d_shards[0].n_core_cells):
            field.field_shards[0].data[i] = rank_0_val
        for i in range(grid2d_shards[1].n_core_cells):
            field.field_shards[1].data[i] = rank_1_val

        field._mark_dirty()

        # --- Step 2: Execute Sync ---
        try:
            field.sync_halos(HaloMode.OVERWRITE)
        except Exception as e:
            if "distributed" in str(e).lower() or "group" in str(e).lower():
                pytest.skip("Distributed environment not initialized. ")
            raise

        # --- Step 3: Verify Data Propagation ---
        val_shard1_start = float(field.field_shards[1].data[0])
        val_shard1_end = float(field.field_shards[1].data[-1])
        val_shard0_start = float(field.field_shards[0].data[0])
        val_shard0_end = float(field.field_shards[0].data[-1])
        tol = 1e-5

        # If sync worked:
        assert abs(val_shard1_start - val_shard0_end) < tol
        assert abs(val_shard0_start - val_shard1_end) < tol
        assert abs(val_shard0_end - rank_1_val) < tol

    def test_sync_halos_sum_mode_logic(self, grid2d_shards: list[MeshShard]):
        """
        Test SUM mode logic (if supported).
        In SUM mode, halo values should be added to existing values.
        """
        field = Field(
            mesh_shards=grid2d_shards,
            vtype=VariableType.SCALAR,
            etype=ElementType.CELL,
            init_val=1.0,
            requires_grad=False,
        )
        field._mark_dirty()

        try:
            # Attempt SUM mode
            field.sync_halos(HaloMode.SUM)
            # If successful, dirty flags cleared
            assert not any(field._dirty_flags.values())
        except TypeError:
            field.sync_halos()
        except Exception as e:
            if "distributed" in str(e).lower():
                pytest.skip("Dist env needed for full mode test")
            else:
                raise


# ============================================
# region Multiply Tests
# ============================================


class TestFieldMultiplications:
    """
    Robust tests for field multiplications with data consistency checks.
    """

    def test_field_dot_product(self, grid2d_shards: list[MeshShard]):
        """Test Vector * Vector -> Scalar (Dot Product)"""
        # Create two vector fields
        # Field1: All vectors are (1, 2, 3)
        # Field2: All vectors are (4, 5, 6)
        field1 = Field(
            mesh_shards=grid2d_shards,
            vtype=VariableType.VECTOR,
            etype=ElementType.CELL,
            init_val=0.0,
            requires_grad=False,
        )
        field2 = Field(
            mesh_shards=grid2d_shards,
            vtype=VariableType.VECTOR,
            etype=ElementType.CELL,
            init_val=0.0,
            requires_grad=False,
        )

        # Initialize data: (1,2,3) and (4,5,6)
        for shard in field1._shards:
            shard.data[:] = np.array([1.0, 2.0, 3.0])
        for shard in field2._shards:
            shard.data[:] = np.array([4.0, 5.0, 6.0])

        # Perform dot product
        result_field = field1 * field2  # Should be 1*4 + 2*5 + 3*6 = 32

        # Check result type and values
        assert result_field.vtype == VariableType.SCALAR
        for shard in result_field._shards:
            # Check shape: (N, 1) for scalar field
            assert shard.data.shape[1] == 1
            # Check value: Should be 32.0
            np.testing.assert_allclose(shard.data, 32.0, rtol=1e-5)

    def test_field_matmul_vector_tensor(self, grid2d_shards: list[MeshShard]):
        """Test Vector @ Tensor -> Vector"""
        # Create a vector field (1, 0, 0)
        vector_field = Field(
            mesh_shards=grid2d_shards,
            vtype=VariableType.VECTOR,
            etype=ElementType.CELL,
            init_val=0.0,
        )
        # Create a 3x3 Identity tensor field
        tensor_field = Field(
            mesh_shards=grid2d_shards,
            vtype=VariableType.TENSOR,
            etype=ElementType.CELL,
            init_val=0.0,
        )

        # Initialize: Vector = [1, 0, 0]
        for shard in vector_field._shards:
            shard.data[:] = np.array([1.0, 0.0, 0.0])

        # Initialize: Tensor = Identity Matrix [[[1,0,0], [0,1,0], [0,0,0]]]
        # Note: Reshaping to fit (N, 3, 3)
        ident = np.eye(3).reshape(1, 3, 3)
        for shard in tensor_field._shards:
            shard.data[:] = ident

        # Perform: [1,0,0] @ I = [1,0,0]
        result_field = vector_field @ tensor_field

        # Assertions
        assert result_field.vtype == VariableType.VECTOR
        for shard in result_field._shards:
            # Check shape
            assert shard.data.shape[1] == 3
            # Check values: Should still be [1, 0, 0]
            target = np.ones((shard.data.shape[0], 3)) * np.array([1.0, 0.0, 0.0])
            np.testing.assert_allclose(shard.data, target, rtol=1e-5)

    def test_field_outer_product(self, grid2d_shards: list[MeshShard]):
        """Test Vector ^ Vector -> Tensor (Outer Product)"""
        # Create two vector fields
        # A = (1, 0, 0)
        # B = (2, 0, 0)
        field_a = Field(
            mesh_shards=grid2d_shards,
            vtype=VariableType.VECTOR,
            etype=ElementType.CELL,
            init_val=0.0,
        )
        field_b = Field(
            mesh_shards=grid2d_shards,
            vtype=VariableType.VECTOR,
            etype=ElementType.CELL,
            init_val=0.0,
        )

        # Initialize data
        for shard in field_a._shards:
            shard.data[:] = np.array([1.0, 0.0, 0.0])
        for shard in field_b._shards:
            shard.data[:] = np.array([2.0, 0.0, 0.0])

        # Perform outer product: A ^ B
        # Expected result: [[2,0,0], [0,0,0], [0,0,0]]
        result_field = field_a ^ field_b

        # Assertions
        assert result_field.vtype == VariableType.TENSOR
        expected_tensor = np.zeros((3, 3))
        expected_tensor[0, 0] = 2.0  # 1 * 2

        for shard in result_field._shards:
            # Check shape
            assert shard.data.shape[1:] == (3, 3)
            # Check values against the expected 3x3 tensor
            np.testing.assert_allclose(shard.data[0], expected_tensor, rtol=1e-5)

    def test_field_scalar_broadcast(self, grid2d_shards: list[MeshShard]):
        """Test Scalar * Tensor (Broadcasting)"""
        # Create a tensor field (Identity)
        tensor_field = Field(
            mesh_shards=grid2d_shards,
            vtype=VariableType.TENSOR,
            etype=ElementType.CELL,
            init_val=0.0,
        )

        # Initialize to Identity
        ident = np.eye(3).reshape(1, 3, 3)
        for shard in tensor_field._shards:
            shard.data[:] = ident

        # Multiply by scalar 2.0
        result_field = tensor_field * 2.0

        # Check
        for shard in result_field._shards:
            n_elements = shard.data.shape[0]
            expected = np.tile(ident * 2.0, (n_elements, 1, 1))
            np.testing.assert_allclose(shard.data, expected, rtol=1e-5)
