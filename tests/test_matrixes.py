# -*- encoding: utf-8 -*-
"""
Unittests for matrixes module.
"""

import pytest
import torch
import numpy as np
import scipy.sparse as sp
from yunmeng.numerics.linalgs import Matrix, TorchMatrix, NumpyMatrix

# ============================================
# region Fixtures
# ============================================


@pytest.fixture
def sample_sparse_data():
    """Create sample sparse matrix data for testing."""
    # 5x5 sparse matrix with 6 non-zero elements
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    rows = np.array([0, 1, 2, 3, 4, 0])
    cols = np.array([0, 1, 2, 3, 4, 1])
    shape = (5, 5)
    return values, rows, cols, shape


@pytest.fixture
def sample_dense_data():
    """Create sample dense matrix data for testing."""
    return np.array([[1.0, 2.0, 0.0], [0.0, 3.0, 4.0], [5.0, 0.0, 6.0]])


@pytest.fixture
def sample_vector():
    """Create sample vector for matrix-vector multiplication."""
    return np.array([1.0, 2.0, 3.0])


# ============================================
# region TorchMatrix
# ============================================


class TestTorchMatrix:
    """Test suite for TorchMatrix class."""

    # ========================================
    # Factory Methods Tests
    # ========================================

    def test_from_data_coo_format(self, sample_sparse_data):
        """Test creating TorchMatrix from COO format data."""
        values, rows, cols, shape = sample_sparse_data
        matrix = TorchMatrix.from_data(values, indices=(rows, cols), shape=shape)

        assert isinstance(matrix, TorchMatrix)
        assert isinstance(matrix, Matrix)
        assert matrix.shape == shape
        assert matrix.nnz == len(values)

    def test_from_data_dense_format(self, sample_dense_data):
        """Test creating TorchMatrix from dense data."""
        matrix = TorchMatrix.from_data(sample_dense_data)

        assert isinstance(matrix, TorchMatrix)
        assert matrix.shape == sample_dense_data.shape
        # Count non-zeros in dense matrix
        expected_nnz = np.count_nonzero(sample_dense_data)
        assert matrix.nnz == expected_nnz

    def test_from_coo(self, sample_sparse_data):
        """Test creating TorchMatrix using from_coo factory method."""
        values, rows, cols, shape = sample_sparse_data
        matrix = TorchMatrix.from_coo(shape, values, rows, cols)

        assert matrix.shape == shape
        assert matrix.nnz == len(values)

    def test_from_csr(self):
        """Test creating TorchMatrix using from_csr factory method."""
        shape = (3, 3)
        values = torch.tensor([1.0, 2.0, 3.0, 4.0])
        ptrs = torch.tensor([0, 2, 3, 4])
        idxs = torch.tensor([0, 1, 2, 0])

        matrix = TorchMatrix.from_csr(shape, values, ptrs, idxs)

        assert matrix.shape == shape
        # For CSR tensors, we check the values directly instead of nnz
        # due to limitations in PyTorch's CSR tensor support
        assert len(matrix._data.values()) == len(values)

    def test_zeros(self):
        """Test creating zero TorchMatrix."""
        shape = (5, 5)
        matrix = TorchMatrix.zeros(shape)

        assert matrix.shape == shape
        assert matrix.nnz == 0

    def test_identity(self):
        """Test creating identity TorchMatrix."""
        size = 4
        matrix = TorchMatrix.identity(size)

        assert matrix.shape == (size, size)
        assert matrix.nnz == size
        # Check diagonal elements are all 1
        diag = matrix.diagonal()
        assert torch.all(diag == 1.0)

    # ========================================
    # Properties Tests
    # ========================================

    def test_shape_property(self, sample_sparse_data):
        """Test shape property of TorchMatrix."""
        values, rows, cols, shape = sample_sparse_data
        matrix = TorchMatrix.from_data(values, indices=(rows, cols), shape=shape)

        assert matrix.shape == shape

    def test_nnz_property(self, sample_sparse_data):
        """Test nnz property of TorchMatrix."""
        values, rows, cols, shape = sample_sparse_data
        matrix = TorchMatrix.from_data(values, indices=(rows, cols), shape=shape)

        assert matrix.nnz == len(values)

    def test_device_property_cpu(self, sample_sparse_data):
        """Test device property of TorchMatrix on CPU."""
        values, rows, cols, shape = sample_sparse_data
        matrix = TorchMatrix.from_data(
            values, indices=(rows, cols), shape=shape, device=torch.device("cpu")
        )

        assert matrix.device.type == "cpu"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_device_property_cuda(self, sample_sparse_data):
        """Test device property of TorchMatrix on CUDA."""
        values, rows, cols, shape = sample_sparse_data
        matrix = TorchMatrix.from_data(
            values, indices=(rows, cols), shape=shape, device=torch.device("cuda:0")
        )

        assert matrix.device.type == "cuda"

    def test_transpose_property(self, sample_sparse_data):
        """Test transpose property of TorchMatrix."""
        values, rows, cols, shape = sample_sparse_data
        matrix = TorchMatrix.from_data(values, indices=(rows, cols), shape=shape)
        transposed = matrix.T

        assert transposed.shape == (shape[1], shape[0])
        assert transposed.nnz == matrix.nnz

    def test_diags_property(self, sample_sparse_data):
        """Test diags property of TorchMatrix."""
        values, rows, cols, shape = sample_sparse_data
        matrix = TorchMatrix.from_data(values, indices=(rows, cols), shape=shape)
        diags = matrix.diags

        assert len(diags) == 1
        # Check diagonal elements
        expected_diag = []
        for i in range(min(shape)):
            for j, (r, c) in enumerate(zip(rows, cols)):
                if r == c == i:
                    expected_diag.append(values[j])
        # Convert to tensor for comparison
        if expected_diag:
            assert torch.allclose(diags[0], torch.tensor(expected_diag))

    # ========================================
    # Utils Tests
    # ========================================

    def test_to_device(self, sample_sparse_data):
        """Test moving TorchMatrix to different device."""
        values, rows, cols, shape = sample_sparse_data
        matrix = TorchMatrix.from_data(values, indices=(rows, cols), shape=shape)

        # Test moving to CPU
        cpu_matrix = matrix.to(torch.device("cpu"))
        assert cpu_matrix.device.type == "cpu"
        assert cpu_matrix.shape == matrix.shape

    def test_convert_coo_to_csr(self, sample_sparse_data):
        """Test converting TorchMatrix from COO to CSR format."""
        values, rows, cols, shape = sample_sparse_data
        matrix = TorchMatrix.from_data(values, indices=(rows, cols), shape=shape)

        csr_matrix = matrix.convert("csr")
        assert csr_matrix._data.layout == torch.sparse_csr
        assert csr_matrix.shape == matrix.shape

    def test_convert_csr_to_coo(self):
        """Test converting TorchMatrix from CSR to COO format."""
        shape = (3, 3)
        values = torch.tensor([1.0, 2.0, 3.0, 4.0])
        ptrs = torch.tensor([0, 2, 3, 4])
        idxs = torch.tensor([0, 1, 2, 0])

        matrix = TorchMatrix.from_csr(shape, values, ptrs, idxs)
        coo_matrix = matrix.convert("coo")

        assert coo_matrix._data.layout == torch.sparse_coo
        assert coo_matrix.shape == matrix.shape

    def test_diagonal(self, sample_sparse_data):
        """Test diagonal extraction from TorchMatrix."""
        values, rows, cols, shape = sample_sparse_data
        matrix = TorchMatrix.from_data(values, indices=(rows, cols), shape=shape)

        # Test main diagonal (offset=0)
        main_diag = matrix.diagonal(0)
        assert isinstance(main_diag, torch.Tensor)

        # Test upper diagonal (offset=1)
        upper_diag = matrix.diagonal(1)
        assert isinstance(upper_diag, torch.Tensor)

        # Test lower diagonal (offset=-1)
        lower_diag = matrix.diagonal(-1)
        assert isinstance(lower_diag, torch.Tensor)

    def test_to_numpy(self, sample_sparse_data):
        """Test converting TorchMatrix to NumpyMatrix."""
        values, rows, cols, shape = sample_sparse_data
        torch_matrix = TorchMatrix.from_data(values, indices=(rows, cols), shape=shape)
        numpy_matrix = torch_matrix.to_numpy()

        assert isinstance(numpy_matrix, NumpyMatrix)
        assert numpy_matrix.shape == torch_matrix.shape
        assert numpy_matrix.nnz == torch_matrix.nnz

    # ========================================
    # Operators Tests
    # ========================================

    def test_matmul_matrix_matrix(self, sample_dense_data):
        """Test matrix-matrix multiplication with TorchMatrix."""
        matrix1 = TorchMatrix.from_data(sample_dense_data)
        matrix2 = TorchMatrix.from_data(sample_dense_data.T)
        result = matrix1 @ matrix2

        assert isinstance(result, TorchMatrix)
        assert result.shape == (sample_dense_data.shape[0], sample_dense_data.shape[0])

    def test_matmul_matrix_vector(self, sample_dense_data, sample_vector):
        """Test matrix-vector multiplication with TorchMatrix."""
        matrix = TorchMatrix.from_data(sample_dense_data)
        vector = torch.tensor(sample_vector)
        result = matrix @ vector

        assert isinstance(result, torch.Tensor)
        assert result.shape[0] == sample_dense_data.shape[0]

    def test_addition(self, sample_sparse_data):
        """Test matrix addition with TorchMatrix."""
        values, rows, cols, shape = sample_sparse_data
        matrix1 = TorchMatrix.from_data(values, indices=(rows, cols), shape=shape)
        matrix2 = TorchMatrix.from_data(values, indices=(rows, cols), shape=shape)
        result = matrix1 + matrix2

        assert isinstance(result, TorchMatrix)
        assert result.shape == shape

    def test_subtraction(self, sample_sparse_data):
        """Test matrix subtraction with TorchMatrix."""
        values, rows, cols, shape = sample_sparse_data
        matrix1 = TorchMatrix.from_data(values, indices=(rows, cols), shape=shape)
        matrix2 = TorchMatrix.from_data(values, indices=(rows, cols), shape=shape)
        result = matrix1 - matrix2

        assert isinstance(result, TorchMatrix)
        assert result.shape == shape
        # Check that all values are zero (sparse matrices may keep zero values)
        result_dense = result._data.to_dense()
        assert torch.allclose(result_dense, torch.zeros_like(result_dense))

    def test_scalar_multiplication(self, sample_sparse_data):
        """Test scalar multiplication with TorchMatrix."""
        values, rows, cols, shape = sample_sparse_data
        matrix = TorchMatrix.from_data(values, indices=(rows, cols), shape=shape)
        scalar = 2.5
        result = matrix * scalar

        assert isinstance(result, TorchMatrix)
        assert result.shape == shape
        assert result.nnz == matrix.nnz

    def test_right_scalar_multiplication(self, sample_sparse_data):
        """Test right scalar multiplication with TorchMatrix."""
        values, rows, cols, shape = sample_sparse_data
        matrix = TorchMatrix.from_data(values, indices=(rows, cols), shape=shape)
        scalar = 2.5
        result = scalar * matrix

        assert isinstance(result, TorchMatrix)
        assert result.shape == shape
        assert result.nnz == matrix.nnz


# ============================================
# region NumpyMatrix
# ============================================


class TestNumpyMatrix:
    """Test suite for NumpyMatrix class."""

    # ========================================
    # Factory Methods Tests
    # ========================================

    def test_from_data_coo_format(self, sample_sparse_data):
        """Test creating NumpyMatrix from COO format data."""
        values, rows, cols, shape = sample_sparse_data
        matrix = NumpyMatrix.from_data(values, indices=(rows, cols), shape=shape)

        assert isinstance(matrix, NumpyMatrix)
        assert isinstance(matrix, Matrix)
        assert matrix.shape == shape
        assert matrix.nnz == len(values)

    def test_from_data_dense_format(self, sample_dense_data):
        """Test creating NumpyMatrix from dense data."""
        matrix = NumpyMatrix.from_data(sample_dense_data)

        assert isinstance(matrix, NumpyMatrix)
        assert matrix.shape == sample_dense_data.shape
        # Count non-zeros in dense matrix
        expected_nnz = np.count_nonzero(sample_dense_data)
        assert matrix.nnz == expected_nnz

    def test_zeros(self):
        """Test creating zero NumpyMatrix."""
        shape = (5, 5)
        matrix = NumpyMatrix.zeros(shape)

        assert matrix.shape == shape
        assert matrix.nnz == 0

    def test_identity(self):
        """Test creating identity NumpyMatrix."""
        size = 4
        matrix = NumpyMatrix.identity(size)

        assert matrix.shape == (size, size)
        assert matrix.nnz == size
        # Check diagonal elements are all 1
        diag = matrix.diagonal()
        assert np.allclose(diag, np.ones(size))

    # ========================================
    # Properties Tests
    # ========================================

    def test_shape_property(self, sample_sparse_data):
        """Test shape property of NumpyMatrix."""
        values, rows, cols, shape = sample_sparse_data
        matrix = NumpyMatrix.from_data(values, indices=(rows, cols), shape=shape)

        assert matrix.shape == shape

    def test_nnz_property(self, sample_sparse_data):
        """Test nnz property of NumpyMatrix."""
        values, rows, cols, shape = sample_sparse_data
        matrix = NumpyMatrix.from_data(values, indices=(rows, cols), shape=shape)

        assert matrix.nnz == len(values)

    def test_device_property(self):
        """Test device property of NumpyMatrix."""
        matrix = NumpyMatrix.zeros((3, 3))

        assert matrix.device == "cpu"

    def test_transpose_property(self, sample_sparse_data):
        """Test transpose property of NumpyMatrix."""
        values, rows, cols, shape = sample_sparse_data
        matrix = NumpyMatrix.from_data(values, indices=(rows, cols), shape=shape)
        transposed = matrix.T

        assert transposed.shape == (shape[1], shape[0])
        assert transposed.nnz == matrix.nnz

    def test_diags_property(self, sample_sparse_data):
        """Test diags property of NumpyMatrix."""
        values, rows, cols, shape = sample_sparse_data
        matrix = NumpyMatrix.from_data(values, indices=(rows, cols), shape=shape)
        diags = matrix.diags

        assert len(diags) == 1
        assert isinstance(diags[0], np.ndarray)

    # ========================================
    # Utils Tests
    # ========================================

    def test_to_device_cpu(self):
        """Test moving NumpyMatrix to CPU device."""
        matrix = NumpyMatrix.zeros((3, 3))
        cpu_matrix = matrix.to("cpu")

        assert cpu_matrix is matrix  # Should return self

    def test_to_device_gpu_raises_error(self):
        """Test that moving NumpyMatrix to GPU raises an error."""
        matrix = NumpyMatrix.zeros((3, 3))

        with pytest.raises(ValueError, match="NumpyMatrix cannot move to GPU"):
            matrix.to("cuda")

    def test_diagonal(self, sample_sparse_data):
        """Test diagonal extraction from NumpyMatrix."""
        values, rows, cols, shape = sample_sparse_data
        matrix = NumpyMatrix.from_data(values, indices=(rows, cols), shape=shape)

        # Test main diagonal (offset=0)
        main_diag = matrix.diagonal(0)
        assert isinstance(main_diag, np.ndarray)

        # Test upper diagonal (offset=1)
        upper_diag = matrix.diagonal(1)
        assert isinstance(upper_diag, np.ndarray)

        # Test lower diagonal (offset=-1)
        lower_diag = matrix.diagonal(-1)
        assert isinstance(lower_diag, np.ndarray)

    def test_to_torch(self, sample_sparse_data):
        """Test converting NumpyMatrix to TorchMatrix."""
        values, rows, cols, shape = sample_sparse_data
        numpy_matrix = NumpyMatrix.from_data(values, indices=(rows, cols), shape=shape)
        torch_matrix = numpy_matrix.to_torch()

        assert isinstance(torch_matrix, TorchMatrix)
        assert torch_matrix.shape == numpy_matrix.shape
        assert torch_matrix.nnz == numpy_matrix.nnz

    # ========================================
    # Operators Tests
    # ========================================

    def test_matmul_matrix_matrix(self, sample_dense_data):
        """Test matrix-matrix multiplication with NumpyMatrix."""
        matrix1 = NumpyMatrix.from_data(sample_dense_data)
        matrix2 = NumpyMatrix.from_data(sample_dense_data.T)
        result = matrix1 @ matrix2

        assert isinstance(result, NumpyMatrix)
        assert result.shape == (sample_dense_data.shape[0], sample_dense_data.shape[0])

    def test_matmul_matrix_vector(self, sample_dense_data, sample_vector):
        """Test matrix-vector multiplication with NumpyMatrix."""
        matrix = NumpyMatrix.from_data(sample_dense_data)
        result = matrix @ sample_vector

        assert isinstance(result, np.ndarray)
        assert result.shape[0] == sample_dense_data.shape[0]

    def test_addition(self, sample_sparse_data):
        """Test matrix addition with NumpyMatrix."""
        values, rows, cols, shape = sample_sparse_data
        matrix1 = NumpyMatrix.from_data(values, indices=(rows, cols), shape=shape)
        matrix2 = NumpyMatrix.from_data(values, indices=(rows, cols), shape=shape)
        result = matrix1 + matrix2

        assert isinstance(result, NumpyMatrix)
        assert result.shape == shape

    def test_subtraction(self, sample_sparse_data):
        """Test matrix subtraction with NumpyMatrix."""
        values, rows, cols, shape = sample_sparse_data
        matrix1 = NumpyMatrix.from_data(values, indices=(rows, cols), shape=shape)
        matrix2 = NumpyMatrix.from_data(values, indices=(rows, cols), shape=shape)
        result = matrix1 - matrix2

        assert isinstance(result, NumpyMatrix)
        assert result.shape == shape
        # Result should be zero matrix
        assert result.nnz == 0

    def test_scalar_multiplication(self, sample_sparse_data):
        """Test scalar multiplication with NumpyMatrix."""
        values, rows, cols, shape = sample_sparse_data
        matrix = NumpyMatrix.from_data(values, indices=(rows, cols), shape=shape)
        scalar = 2.5
        result = matrix * scalar

        assert isinstance(result, NumpyMatrix)
        assert result.shape == shape
        assert result.nnz == matrix.nnz

    def test_right_scalar_multiplication(self, sample_sparse_data):
        """Test right scalar multiplication with NumpyMatrix."""
        values, rows, cols, shape = sample_sparse_data
        matrix = NumpyMatrix.from_data(values, indices=(rows, cols), shape=shape)
        scalar = 2.5
        result = scalar * matrix

        assert isinstance(result, NumpyMatrix)
        assert result.shape == shape
        assert result.nnz == matrix.nnz


# ============================================
# Cross-Implementation Tests
# ============================================


class TestCrossImplementation:
    """Test suite for cross-implementation compatibility."""

    def test_torch_to_numpy_conversion(self, sample_sparse_data):
        """Test converting TorchMatrix to NumpyMatrix."""
        values, rows, cols, shape = sample_sparse_data
        torch_matrix = TorchMatrix.from_data(values, indices=(rows, cols), shape=shape)
        numpy_matrix = torch_matrix.to_numpy()

        assert isinstance(numpy_matrix, NumpyMatrix)
        assert numpy_matrix.shape == torch_matrix.shape
        assert numpy_matrix.nnz == torch_matrix.nnz

    def test_numpy_to_torch_conversion(self, sample_sparse_data):
        """Test converting NumpyMatrix to TorchMatrix."""
        values, rows, cols, shape = sample_sparse_data
        numpy_matrix = NumpyMatrix.from_data(values, indices=(rows, cols), shape=shape)
        torch_matrix = numpy_matrix.to_torch()

        assert isinstance(torch_matrix, TorchMatrix)
        assert torch_matrix.shape == numpy_matrix.shape
        assert torch_matrix.nnz == numpy_matrix.nnz

    def test_round_trip_conversion(self, sample_sparse_data):
        """Test round-trip conversion between TorchMatrix and NumpyMatrix."""
        values, rows, cols, shape = sample_sparse_data

        # Torch -> Numpy -> Torch
        torch_matrix1 = TorchMatrix.from_data(values, indices=(rows, cols), shape=shape)
        numpy_matrix = torch_matrix1.to_numpy()
        torch_matrix2 = numpy_matrix.to_torch()

        assert torch_matrix1.shape == torch_matrix2.shape
        assert torch_matrix1.nnz == torch_matrix2.nnz

        # Numpy -> Torch -> Numpy
        numpy_matrix1 = NumpyMatrix.from_data(values, indices=(rows, cols), shape=shape)
        torch_matrix = numpy_matrix1.to_torch()
        numpy_matrix2 = torch_matrix.to_numpy()

        assert numpy_matrix1.shape == numpy_matrix2.shape
        assert numpy_matrix1.nnz == numpy_matrix2.nnz
