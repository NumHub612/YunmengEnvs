# -*- encoding: utf-8 -*-
"""
Unit tests for LinearEqs class.
"""
import pytest
import numpy as np
import torch
import scipy.sparse as sp

from core.numerics.mats.linalgs import LinearEqs
from core.numerics.mats.sparse import TorchMatrix, NumpyMatrix
from core.numerics.fields.fields import Field
from core.numerics.enums import VariableType, ElementType, BackendType
from core.numerics.algos.parts import MeshShard


# ============================================
# region Fixtures
# ============================================


@pytest.fixture
def simple_mesh_shard():
    """Create a simple mesh shard for testing."""
    return MeshShard.from_size(5, ElementType.CELL)


@pytest.fixture
def sample_torch_matrix():
    """Create a sample TorchMatrix for testing."""
    # 5x5 symmetric positive definite matrix
    data = np.array(
        [
            [4.0, 1.0, 0.0, 0.0, 0.0],
            [1.0, 4.0, 1.0, 0.0, 0.0],
            [0.0, 1.0, 4.0, 1.0, 0.0],
            [0.0, 0.0, 1.0, 4.0, 1.0],
            [0.0, 0.0, 0.0, 1.0, 4.0],
        ]
    )
    return TorchMatrix.from_data(data)


@pytest.fixture
def sample_numpy_matrix():
    """Create a sample NumpyMatrix for testing."""
    # 5x5 symmetric positive definite matrix
    data = np.array(
        [
            [4.0, 1.0, 0.0, 0.0, 0.0],
            [1.0, 4.0, 1.0, 0.0, 0.0],
            [0.0, 1.0, 4.0, 1.0, 0.0],
            [0.0, 0.0, 1.0, 4.0, 1.0],
            [0.0, 0.0, 0.0, 1.0, 4.0],
        ]
    )
    return NumpyMatrix.from_data(data)


@pytest.fixture
def sample_rhs_field(simple_mesh_shard):
    """Create a sample RHS field for testing."""
    # Create a scalar field with size 5
    return Field.from_size(
        size=5,
        vtype=VariableType.SCALAR,
        etype=ElementType.CELL,
        init_val=1.0,
    )


@pytest.fixture
def sample_vector_rhs_field(simple_mesh_shard):
    """Create a sample vector RHS field for testing."""
    # Create a vector field with size 5 and 3 components
    return Field.from_size(
        size=5,
        vtype=VariableType.VECTOR,
        etype=ElementType.CELL,
        init_val=1.0,
    )


# ============================================
# region Initialization Tests
# ============================================


class TestLinearEqsInitialization:
    """Test suite for LinearEqs initialization."""

    def test_init_torch_matrix_scalar_rhs(self, sample_torch_matrix, sample_rhs_field):
        """Test initialization with TorchMatrix and scalar RHS field."""
        eqs = LinearEqs(sample_torch_matrix, sample_rhs_field)

        assert eqs.size == 5
        assert eqs.matrix is sample_torch_matrix
        assert eqs.rhs is sample_rhs_field

    def test_init_numpy_matrix_scalar_rhs(self, sample_numpy_matrix, sample_rhs_field):
        """Test initialization with NumpyMatrix and scalar RHS field."""
        eqs = LinearEqs(sample_numpy_matrix, sample_rhs_field)

        assert eqs.size == 5
        assert eqs.matrix is sample_numpy_matrix
        assert eqs.rhs is sample_rhs_field

    def test_init_non_square_matrix(self, sample_rhs_field):
        """Test that initialization raises ValueError for non-square matrix."""
        # Create a non-square matrix (5x3)
        data = np.ones((5, 3))
        matrix = TorchMatrix.from_data(data)

        with pytest.raises(ValueError, match="not square"):
            LinearEqs(matrix, sample_rhs_field)

    def test_init_incompatible_size(self, sample_rhs_field):
        """Test that initialization raises ValueError for incompatible sizes."""
        # Create a 3x3 matrix (incompatible with RHS size of 5)
        data = np.eye(3)
        matrix = TorchMatrix.from_data(data)

        with pytest.raises(ValueError, match="not compatible"):
            LinearEqs(matrix, sample_rhs_field)


# ============================================
# region Property Tests
# ============================================


class TestLinearEqsProperties:
    """Test suite for LinearEqs properties."""

    def test_size_property(self, sample_torch_matrix, sample_rhs_field):
        """Test size property."""
        eqs = LinearEqs(sample_torch_matrix, sample_rhs_field)

        assert eqs.size == 5

    def test_matrix_property(self, sample_torch_matrix, sample_rhs_field):
        """Test matrix property."""
        eqs = LinearEqs(sample_torch_matrix, sample_rhs_field)

        assert eqs.matrix is sample_torch_matrix

    def test_rhs_property(self, sample_torch_matrix, sample_rhs_field):
        """Test rhs property."""
        eqs = LinearEqs(sample_torch_matrix, sample_rhs_field)

        assert eqs.rhs is sample_rhs_field


# ============================================
# region Operations Tests
# ============================================


class TestLinearEqsOperations:
    """Test suite for LinearEqs operations."""

    def test_addition(self, sample_torch_matrix, sample_rhs_field):
        """Test addition of two LinearEqs."""
        eqs1 = LinearEqs(sample_torch_matrix, sample_rhs_field)
        eqs2 = LinearEqs(sample_torch_matrix, sample_rhs_field)

        result = eqs1 + eqs2

        assert isinstance(result, LinearEqs)
        assert result.size == 5

    def test_in_place_addition(self, sample_torch_matrix, sample_rhs_field):
        """Test in-place addition of LinearEqs."""
        eqs1 = LinearEqs(sample_torch_matrix, sample_rhs_field)
        eqs2 = LinearEqs(sample_torch_matrix, sample_rhs_field)

        eqs1 += eqs2

    def test_subtraction(self, sample_torch_matrix, sample_rhs_field):
        """Test subtraction of two LinearEqs."""
        eqs1 = LinearEqs(sample_torch_matrix, sample_rhs_field)
        eqs2 = LinearEqs(sample_torch_matrix, sample_rhs_field)

        result = eqs1 - eqs2

        assert isinstance(result, LinearEqs)
        assert result.size == 5

    def test_in_place_subtraction(self, sample_torch_matrix, sample_rhs_field):
        """Test in-place subtraction of LinearEqs."""
        eqs1 = LinearEqs(sample_torch_matrix, sample_rhs_field)
        eqs2 = LinearEqs(sample_torch_matrix, sample_rhs_field)

        eqs1 -= eqs2

    def test_addition_incompatible_size(self, sample_torch_matrix, sample_rhs_field):
        """Test that addition raises ValueError for incompatible sizes."""
        eqs1 = LinearEqs(sample_torch_matrix, sample_rhs_field)

        # Create a smaller LinearEqs (3x3 matrix)
        data = np.eye(3)
        matrix = TorchMatrix.from_data(data)
        small_field = Field.from_size(
            size=3,
            vtype=VariableType.SCALAR,
            etype=ElementType.CELL,
            init_val=1.0,
        )
        eqs2 = LinearEqs(matrix, small_field)

        with pytest.raises(ValueError, match="different sizes"):
            _ = eqs1 + eqs2

    def test_addition_incompatible_type(self, sample_torch_matrix, sample_rhs_field):
        """Test that addition raises ValueError for incompatible types."""
        eqs1 = LinearEqs(sample_torch_matrix, sample_rhs_field)

        with pytest.raises(ValueError, match="Require LinearEqs type"):
            _ = eqs1 + 1.0


# ============================================
# region Scalarize Tests
# ============================================


class TestLinearEqsScalarize:
    """Test suite for LinearEqs scalarize operation."""

    def test_scalarize_scalar_field(self, sample_torch_matrix, sample_rhs_field):
        """Test scalarizing a scalar field (should return a single LinearEqs in list)."""
        eqs = LinearEqs(sample_torch_matrix, sample_rhs_field)

        scalar_eqs = eqs.scalarize()

        assert len(scalar_eqs) == 1
        assert all(isinstance(e, LinearEqs) for e in scalar_eqs)
        assert all(e.size == 5 for e in scalar_eqs)

    def test_scalarize_vector_field(self, sample_torch_matrix, sample_vector_rhs_field):
        """Test scalarizing a vector field (should return multiple LinearEqs in list)."""
        eqs = LinearEqs(sample_torch_matrix, sample_vector_rhs_field)

        scalar_eqs = eqs.scalarize()

        # VECTOR3D has 3 components
        assert len(scalar_eqs) == 3
        assert all(isinstance(e, LinearEqs) for e in scalar_eqs)
        assert all(e.size == 5 for e in scalar_eqs)


# ============================================
# region Solve Tests
# ============================================


class TestLinearEqsSolve:
    """Test suite for LinearEqs solve operation."""

    def test_solve_torch_matrix_scalar_rhs(self, sample_torch_matrix, sample_rhs_field):
        """Test solving linear equations with TorchMatrix and scalar RHS."""
        # Set RHS to known values
        for i in range(sample_rhs_field.size):
            sample_rhs_field[i] = float(i + 1)

        eqs = LinearEqs(sample_torch_matrix, sample_rhs_field)
        solution = eqs.solve()

        # Verify solution
        assert isinstance(solution, Field)
        assert solution.size == 5
        assert solution.vtype == VariableType.SCALAR

        # Verify that solution satisfies the equations (A * x = b)
        rhs_array = sample_rhs_field.gather_to_host()
        solution_array = solution.gather_to_host()

        # Compute A * x
        mat_dense = sample_torch_matrix.data.to_dense()
        computed_rhs = (mat_dense @ torch.tensor(solution_array)).numpy()

        # Check that computed RHS matches actual RHS
        # Ensure both arrays have the same shape
        computed_rhs = computed_rhs.flatten()
        rhs_array_flat = rhs_array.flatten()
        np.testing.assert_allclose(computed_rhs, rhs_array_flat, rtol=1e-6)

    def test_solve_numpy_matrix_scalar_rhs(self, sample_numpy_matrix, sample_rhs_field):
        """Test solving linear equations with NumpyMatrix and scalar RHS."""
        # Set RHS to known values
        for i in range(sample_rhs_field.size):
            sample_rhs_field[i] = float(i + 1)

        eqs = LinearEqs(sample_numpy_matrix, sample_rhs_field)
        solution = eqs.solve()

        # Verify solution
        assert isinstance(solution, Field)
        assert solution.size == 5
        assert solution.vtype == VariableType.SCALAR

        # Verify that solution satisfies the equations (A * x = b)
        rhs_array = sample_rhs_field.gather_to_host()
        solution_array = solution.gather_to_host()

        # Compute A * x
        computed_rhs = sample_numpy_matrix.data @ solution_array

        # Check that computed RHS matches actual RHS
        np.testing.assert_allclose(computed_rhs, rhs_array, rtol=1e-6)

    def test_solve_torch_matrix_vector_rhs(
        self, sample_torch_matrix, sample_vector_rhs_field
    ):
        """Test solving linear equations with TorchMatrix and vector RHS."""
        # Set RHS to known values
        for i in range(sample_vector_rhs_field.size):
            sample_vector_rhs_field[i] = np.array(
                [float(i + 1), float(i + 2), float(i + 3)]
            )

        print("0", sample_vector_rhs_field.shape)
        eqs = LinearEqs(sample_torch_matrix, sample_vector_rhs_field)
        solution = eqs.solve()

        # Verify solution
        assert isinstance(solution, Field)
        assert solution.size == 5
        assert solution.vtype == VariableType.VECTOR

        # Verify that solution satisfies the equations (A * x = b)
        rhs_array = sample_vector_rhs_field.gather_to_host()
        solution_array = solution.gather_to_host()

        # Compute A * x for each component
        mat_dense = sample_torch_matrix.data.to_dense()
        for comp in range(3):
            computed_rhs = (mat_dense @ torch.tensor(solution_array[:, comp])).numpy()
            np.testing.assert_allclose(computed_rhs, rhs_array[:, comp], rtol=1e-6)

    def test_solve_numpy_matrix_vector_rhs(
        self, sample_numpy_matrix, sample_vector_rhs_field
    ):
        """Test solving linear equations with NumpyMatrix and vector RHS."""
        # Set RHS to known values
        for i in range(sample_vector_rhs_field.size):
            sample_vector_rhs_field[i] = np.array(
                [float(i + 1), float(i + 2), float(i + 3)]
            )

        eqs = LinearEqs(sample_numpy_matrix, sample_vector_rhs_field)
        solution = eqs.solve()

        # Verify solution
        assert isinstance(solution, Field)
        assert solution.size == 5
        assert solution.vtype == VariableType.VECTOR

        # Verify that solution satisfies the equations (A * x = b)
        rhs_array = sample_vector_rhs_field.gather_to_host()
        solution_array = solution.gather_to_host()

        # Compute A * x for each component
        for comp in range(3):
            computed_rhs = sample_numpy_matrix.data @ solution_array[:, comp]
            np.testing.assert_allclose(computed_rhs, rhs_array[:, comp], rtol=1e-6)

    def test_solve_identity_matrix(self, sample_rhs_field):
        """Test solving with identity matrix (solution should equal RHS)."""
        # Create identity matrix
        matrix = TorchMatrix.identity(5)
        # Set RHS to known values
        for i in range(sample_rhs_field.size):
            sample_rhs_field[i] = float(i + 1)

        eqs = LinearEqs(matrix, sample_rhs_field)
        solution = eqs.solve()

        # Verify solution equals RHS
        rhs_array = sample_rhs_field.gather_to_host()
        solution_array = solution.gather_to_host()

        np.testing.assert_allclose(solution_array, rhs_array, rtol=1e-6)

    def test_solve_sparse_matrix(self, sample_rhs_field):
        """Test solving with sparse matrix."""
        # Create a sparse tridiagonal matrix
        n = 5
        rows = []
        cols = []
        values = []
        for i in range(n):
            rows.append(i)
            cols.append(i)
            values.append(4.0)  # Diagonal
            if i > 0:
                rows.append(i)
                cols.append(i - 1)
                values.append(1.0)  # Lower diagonal
            if i < n - 1:
                rows.append(i)
                cols.append(i + 1)
                values.append(1.0)  # Upper diagonal

        matrix = TorchMatrix.from_data(
            values=np.array(values),
            indices=(np.array(rows), np.array(cols)),
            shape=(n, n),
        )

        # Set RHS to known values
        for i in range(sample_rhs_field.size):
            sample_rhs_field[i] = float(i + 1)

        eqs = LinearEqs(matrix, sample_rhs_field)
        solution = eqs.solve()

        # Verify solution
        assert isinstance(solution, Field)
        assert solution.size == 5
