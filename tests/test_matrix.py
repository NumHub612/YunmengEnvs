# -*- encoding: utf-8 -*-
import unittest
import torch
import numpy as np
import cupy as cp
import scipy.sparse as sp
from scipy.sparse import dok_matrix
from cupyx.scipy.sparse import coo_matrix
from configs.settings import settings
from core.numerics.fields import VariableType
from core.numerics.matrix import CupyMatrix, TorchMatrix, SciMatrix, SparseMatrix


def check_cuda_env():
    try:
        if not cp.cuda.is_available():
            return False

        cuda_version = cp.cuda.runtime.runtimeGetVersion()
        if cuda_version < 12000:
            return False
    except Exception as e:
        print(f"Error checking CUDA version: {e}")
        return False
    return True


class TestTorchMatrix(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        print(f"\n---------- Testing < {cls.__name__} > \n")

    @classmethod
    def tearDownClass(cls):
        print("\n---------- Done \n")

    def setUp(self):
        self.device = settings.DEVICE
        self.dtype = torch.float64
        if settings.FPTYPE == "fp16":
            self.dtype = torch.float16
        elif settings.FPTYPE == "fp32":
            self.dtype = torch.float32

        self.shape = (3, 3)
        self.indices = torch.tensor(
            [[0, 1, 2], [0, 1, 2]], dtype=torch.long, device=self.device
        )
        self.values = torch.tensor(
            [1.0, 2.0, 3.0], dtype=self.dtype, device=self.device
        )
        self.matrix = TorchMatrix(self.shape, self.indices, self.values, self.device)

    def test_init(self):
        self.assertEqual(self.matrix.shape, self.shape)
        self.assertTrue(torch.equal(self.matrix.data.indices(), self.indices))
        self.assertTrue(torch.equal(self.matrix.data.values(), self.values))

        empty_matrix = TorchMatrix(self.shape)
        self.assertEqual(empty_matrix.shape, self.shape)
        self.assertEqual(empty_matrix.data._nnz(), 0)

    def test_from_data(self):
        matrix_from_data = TorchMatrix.from_data(self.shape, self.indices, self.values)
        self.assertTrue(torch.equal(matrix_from_data.data.indices(), self.indices))
        self.assertTrue(torch.equal(matrix_from_data.data.values(), self.values))

    def test_identity(self):
        identity_matrix = TorchMatrix.identity((3, 3))
        expected_indices = torch.tensor(
            [[0, 1, 2], [0, 1, 2]], dtype=torch.long, device=self.device
        )
        expected_values = torch.ones(3, dtype=self.dtype, device=self.device)
        self.assertTrue(torch.equal(identity_matrix.data.indices(), expected_indices))
        self.assertTrue(torch.equal(identity_matrix.data.values(), expected_values))

    def test_zeros(self):
        zeros_matrix = TorchMatrix.zeros((2, 2))
        self.assertEqual(zeros_matrix.data._nnz(), 0)

    def test_scalarize(self):
        scalarized_matrix = self.matrix.scalarize()
        self.assertEqual(len(scalarized_matrix), 1)
        self.assertTrue(torch.equal(scalarized_matrix[0].data.indices(), self.indices))
        self.assertTrue(torch.equal(scalarized_matrix[0].data.values(), self.values))

    def test_to_dense(self):
        dense_matrix = self.matrix.to_dense()
        expected_dense = np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]])
        np.testing.assert_array_equal(dense_matrix, expected_dense)

    def test_properties(self):
        self.assertEqual(self.matrix.shape, self.shape)
        self.assertEqual(self.matrix.dtype, VariableType.SCALAR)
        self.assertEqual(self.matrix.nnz, [3])
        self.assertEqual(np.all(self.matrix.diag[0] == np.array([1.0, 2.0, 3.0])), True)

    def test_transpose(self):
        transposed_matrix = self.matrix.T
        expected_indices = torch.tensor(
            [[0, 1, 2], [0, 1, 2]], dtype=torch.long, device=self.device
        )
        self.assertTrue(torch.equal(transposed_matrix.data.indices(), expected_indices))
        self.assertTrue(torch.equal(transposed_matrix.data.values(), self.values))

    def test_inverse(self):
        dense_matrix = self.matrix.to_dense()
        inv_dense_matrix = np.linalg.inv(dense_matrix)
        inv_matrix = self.matrix.inv
        np.testing.assert_array_almost_equal(inv_matrix.to_dense(), inv_dense_matrix)

    def test_determinant(self):
        dense_matrix = self.matrix.to_dense()
        expected_det = np.linalg.det(dense_matrix)
        self.assertAlmostEqual(self.matrix.det[0], expected_det)

    def test_getitem(self):
        self.assertEqual(self.matrix[0, 0], 1.0)
        self.assertEqual(self.matrix[1, 1], 2.0)
        self.assertEqual(self.matrix[2, 2], 3.0)
        self.assertEqual(self.matrix[0, 1], 0.0)

    def test_setitem(self):
        self.matrix[0, 0] = 5.0
        self.assertEqual(self.matrix[0, 0], 5.0)
        self.matrix[0, 1] = 4.0
        self.assertEqual(self.matrix[0, 1], 4.0)
        self.assertTrue(
            torch.equal(
                self.matrix.data.values(),
                torch.tensor([5.0, 4.0, 2.0, 3.0], device=self.device),
            )
        )

    def test_add(self):
        other_matrix = TorchMatrix(self.shape, self.indices, self.values)
        result_matrix = self.matrix + other_matrix
        expected_values = torch.tensor(
            [2.0, 4.0, 6.0], dtype=self.dtype, device=self.device
        )
        self.assertTrue(torch.equal(result_matrix.data.values(), expected_values))

    def test_sub(self):
        other_matrix = TorchMatrix(self.shape, self.indices, self.values)
        result_matrix = self.matrix - other_matrix
        expected_values = torch.tensor(
            [0.0, 0.0, 0.0], dtype=self.dtype, device=self.device
        )
        self.assertTrue(torch.equal(result_matrix.data.values(), expected_values))

    def test_mul(self):
        other_matrix = TorchMatrix(self.shape, self.indices, self.values)
        result_matrix = self.matrix * other_matrix
        expected_values = torch.tensor(
            [1.0, 4.0, 9.0], dtype=self.dtype, device=self.device
        )
        self.assertTrue(torch.equal(result_matrix.data.values(), expected_values))

    def test_truediv(self):
        result_matrix = self.matrix / 2.0
        expected_values = torch.tensor(
            [0.5, 1.0, 1.5], dtype=self.dtype, device=self.device
        )
        self.assertTrue(torch.equal(result_matrix.data.values(), expected_values))

    def test_neg(self):
        neg_matrix = -self.matrix
        expected_values = torch.tensor(
            [-1.0, -2.0, -3.0], dtype=self.dtype, device=self.device
        )
        self.assertTrue(torch.equal(neg_matrix.data.values(), expected_values))

    def test_abs(self):
        abs_matrix = abs(self.matrix)
        expected_values = torch.tensor(
            [1.0, 2.0, 3.0], dtype=self.dtype, device=self.device
        )
        self.assertTrue(torch.equal(abs_matrix.data.values(), expected_values))

    def peformance_test(self):
        import time

        matrix_size = 10_000_000

        # init
        start_time = time.time()
        matrix = TorchMatrix.identity((matrix_size, matrix_size), self.device)
        end_time = time.time()
        print(f"Init time: {end_time - start_time} s")

        # add
        start_time = time.time()
        mat2 = matrix + matrix
        end_time = time.time()
        print(f"Add time: {end_time - start_time} s")

        # sub
        start_time = time.time()
        mat3 = mat2 - matrix
        end_time = time.time()
        print(f"Sub time: {end_time - start_time} s")

        # mul
        start_time = time.time()
        matrix *= 3.0
        end_time = time.time()
        print(f"Mul time: {end_time - start_time} s")


@unittest.skipIf(not check_cuda_env(), "No CUDA device available")
class TestCupyMatrix(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print(f"\n---------- Testing < {cls.__name__} > \n")

    @classmethod
    def tearDownClass(cls):
        print("\n---------- Done \n")

    def setUp(self):
        self.shape = (3, 3)
        self.row_indices = cp.array([0, 1, 2])
        self.col_indices = cp.array([0, 1, 2])
        self.values = cp.array([1.0, 2.0, 3.0])
        self.cupy_matrix = CupyMatrix(
            self.shape, self.row_indices, self.col_indices, self.values
        )

    def test_init(self):
        self.assertEqual(self.cupy_matrix.shape, self.shape)
        self.assertIsInstance(self.cupy_matrix.data, coo_matrix)
        self.assertEqual(self.cupy_matrix.data.shape, self.shape)
        self.assertTrue(cp.allclose(self.cupy_matrix.data.data, self.values))

    def test_from_matrix(self):
        coo = coo_matrix(
            (self.values, (self.row_indices, self.col_indices)), shape=self.shape
        )
        cupy_matrix_from_coo = CupyMatrix.from_matrix(coo)
        self.assertIsInstance(cupy_matrix_from_coo, CupyMatrix)
        self.assertTrue(cp.allclose(cupy_matrix_from_coo.data.data, self.values))

    def test_from_data(self):
        cupy_matrix_from_data = CupyMatrix.from_data(
            self.shape, (self.row_indices, self.col_indices), self.values
        )
        self.assertIsInstance(cupy_matrix_from_data, CupyMatrix)
        self.assertTrue(cp.allclose(cupy_matrix_from_data.data.data, self.values))

    def test_identity(self):
        identity_matrix = CupyMatrix.identity((3, 3))
        self.assertIsInstance(identity_matrix, CupyMatrix)
        self.assertTrue(cp.allclose(identity_matrix.data.diagonal(), cp.ones(3)))

    def test_zeros(self):
        zero_matrix = CupyMatrix.zeros((3, 3))
        self.assertIsInstance(zero_matrix, CupyMatrix)
        self.assertEqual(zero_matrix.data.nnz, 0)

    def test_properties(self):
        self.assertEqual(self.cupy_matrix.shape, self.shape)
        self.assertEqual(self.cupy_matrix.nnz, [3])
        self.assertTrue(cp.allclose(self.cupy_matrix.diag, [cp.array([1.0, 2.0, 3.0])]))
        self.assertIsInstance(self.cupy_matrix.T, CupyMatrix)
        self.assertIsInstance(self.cupy_matrix.inv, CupyMatrix)
        self.assertIsInstance(self.cupy_matrix.det, list)
        self.assertEqual(len(self.cupy_matrix.det), 1)

    def test_to_dense(self):
        dense_matrix = self.cupy_matrix.to_dense()
        self.assertIsInstance(dense_matrix, np.ndarray)
        self.assertTrue(
            np.allclose(dense_matrix, cp.asnumpy(self.cupy_matrix.data.todense()))
        )

    def test_addition(self):
        other_matrix = CupyMatrix(
            self.shape, self.row_indices, self.col_indices, self.values
        )
        result_matrix = self.cupy_matrix + other_matrix
        self.assertIsInstance(result_matrix, CupyMatrix)
        self.assertTrue(cp.allclose(result_matrix.data.data, self.values * 2))

    def test_subtraction(self):
        other_matrix = CupyMatrix(
            self.shape, self.row_indices, self.col_indices, self.values
        )
        result_matrix = self.cupy_matrix - other_matrix
        self.assertIsInstance(result_matrix, CupyMatrix)
        self.assertTrue(len(result_matrix.data.data) == 0)

    def test_multiplication(self):
        other_matrix = CupyMatrix(
            self.shape, self.row_indices, self.col_indices, self.values
        )
        result_matrix = self.cupy_matrix * other_matrix
        self.assertIsInstance(result_matrix, CupyMatrix)
        self.assertTrue(cp.allclose(result_matrix.data.data, self.values * self.values))

    def test_scalar_operations(self):
        scalar = 2.0
        result_matrix = self.cupy_matrix * scalar
        self.assertIsInstance(result_matrix, CupyMatrix)
        self.assertTrue(cp.allclose(result_matrix.data.data, self.values * scalar))

        result_matrix = self.cupy_matrix / scalar
        self.assertIsInstance(result_matrix, CupyMatrix)
        self.assertTrue(cp.allclose(result_matrix.data.data, self.values / scalar))

    def test_inplace_operations(self):
        scalar = 2.0
        other_matrix = CupyMatrix(
            self.shape, self.row_indices, self.col_indices, self.values
        )
        self.cupy_matrix += other_matrix
        self.assertTrue(cp.allclose(self.cupy_matrix.data.data, self.values * 2))

        self.cupy_matrix -= other_matrix
        self.assertTrue(cp.allclose(self.cupy_matrix.data.data, self.values))

        self.cupy_matrix *= scalar
        self.assertTrue(cp.allclose(self.cupy_matrix.data.data, self.values * scalar))

        self.cupy_matrix /= scalar
        self.assertTrue(cp.allclose(self.cupy_matrix.data.data, self.values))

    def test_transpose(self):
        transposed_matrix = self.cupy_matrix.T
        self.assertIsInstance(transposed_matrix, CupyMatrix)
        self.assertTrue(cp.allclose(transposed_matrix.data.data, self.values))
        self.assertTrue(cp.allclose(transposed_matrix.data.row, self.col_indices))
        self.assertTrue(cp.allclose(transposed_matrix.data.col, self.row_indices))

    def test_inverse(self):
        inv_matrix = self.cupy_matrix.inv
        self.assertIsInstance(inv_matrix, CupyMatrix)
        dense_product = inv_matrix.to_dense() @ self.cupy_matrix.to_dense()
        identity_matrix = np.eye(self.shape[0])
        self.assertTrue(np.allclose(dense_product, identity_matrix))

    def test_determinant(self):
        det = self.cupy_matrix.det
        self.assertIsInstance(det, list)
        self.assertEqual(len(det), 1)
        self.assertAlmostEqual(det[0], cp.linalg.det(self.cupy_matrix.data.todense()))

    def peformance_test(self):
        import time

        matrix_size = 10_000_000

        # init
        start_time = time.time()
        matrix = CupyMatrix.identity((matrix_size, matrix_size))
        end_time = time.time()
        print(f"Init time: {end_time - start_time} s")

        # add
        start_time = time.time()
        mat2 = matrix + matrix
        end_time = time.time()
        print(f"Add time: {end_time - start_time} s")

        # sub
        start_time = time.time()
        mat3 = mat2 - matrix
        end_time = time.time()
        print(f"Sub time: {end_time - start_time} s")

        # mul
        start_time = time.time()
        matrix *= 3.0
        end_time = time.time()
        print(f"Mul time: {end_time - start_time} s")


class TestSciMatrix(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        print(f"\n---------- Testing < {cls.__name__} > \n")

    @classmethod
    def tearDownClass(cls):
        print("\n---------- Done \n")

    def setUp(self):
        self.shape = (3, 3)
        self.row_indices = np.array([0, 1, 2])
        self.col_indices = np.array([0, 1, 2])
        self.values = np.array([1.0, 2.0, 3.0])
        self.sci_matrix = SciMatrix(
            self.shape, self.row_indices, self.col_indices, self.values
        )

    def test_init(self):
        self.assertEqual(self.sci_matrix.shape, self.shape)
        self.assertIsInstance(self.sci_matrix.data, dok_matrix)
        self.assertEqual(self.sci_matrix.data.shape, self.shape)
        values = np.array(list(self.sci_matrix.data.values()))
        self.assertTrue(np.allclose(values, self.values))

    def test_from_matrix(self):
        coo = sp.coo_matrix(
            (self.values, (self.row_indices, self.col_indices)), shape=self.shape
        )
        dok = coo.todok()
        sci_matrix_from_dok = SciMatrix.from_matrix(dok)
        self.assertIsInstance(sci_matrix_from_dok, SciMatrix)

        values = np.array(list(sci_matrix_from_dok.data.values()))
        self.assertTrue(np.allclose(values, self.values))

    def test_from_data(self):
        indices = np.array([self.row_indices, self.col_indices])
        sci_matrix_from_data = SciMatrix.from_data(self.shape, indices, self.values)
        self.assertIsInstance(sci_matrix_from_data, SciMatrix)
        values = np.array(list(sci_matrix_from_data.data.values()))
        self.assertTrue(np.allclose(values, self.values))

    def test_zeros(self):
        zero_matrix = SciMatrix.zeros((3, 3))
        self.assertIsInstance(zero_matrix, SciMatrix)
        self.assertEqual(zero_matrix.data.nnz, 0)

    def test_identity(self):
        identity_matrix = SciMatrix.identity((3, 3))
        self.assertIsInstance(identity_matrix, SciMatrix)
        self.assertTrue(np.allclose(identity_matrix.data.diagonal(), np.ones(3)))

    def test_properties(self):
        self.assertEqual(self.sci_matrix.shape, self.shape)
        self.assertEqual(self.sci_matrix.nnz, [3])
        self.assertTrue(np.allclose(self.sci_matrix.diag, [np.array([1.0, 2.0, 3.0])]))
        self.assertIsInstance(self.sci_matrix.T, SciMatrix)
        self.assertIsInstance(self.sci_matrix.inv, SciMatrix)
        self.assertIsInstance(self.sci_matrix.det, list)
        self.assertEqual(len(self.sci_matrix.det), 1)

    def test_to_dense(self):
        dense_matrix = self.sci_matrix.to_dense()
        self.assertIsInstance(dense_matrix, np.ndarray)
        self.assertTrue(np.allclose(dense_matrix, self.sci_matrix.data.toarray()))

    def test_addition(self):
        other_matrix = SciMatrix(
            self.shape, self.row_indices, self.col_indices, self.values
        )
        result_matrix = self.sci_matrix + other_matrix
        self.assertIsInstance(result_matrix, SciMatrix)
        values = np.array(list(result_matrix.data.values()))
        self.assertTrue(np.allclose(values, self.values * 2))

    def test_subtraction(self):
        other_matrix = SciMatrix(
            self.shape, self.row_indices, self.col_indices, self.values
        )
        result_matrix = self.sci_matrix - other_matrix
        self.assertIsInstance(result_matrix, SciMatrix)
        values = np.array(list(result_matrix.data.values()))
        self.assertTrue(len(values) == 0)

    def test_multiplication(self):
        other_matrix = SciMatrix(
            self.shape, self.row_indices, self.col_indices, self.values
        )
        result_matrix = self.sci_matrix * other_matrix
        self.assertIsInstance(result_matrix, SciMatrix)
        values = np.array(list(result_matrix.data.values()))
        self.assertTrue(np.allclose(values, self.values * self.values))

    def test_scalar_operations(self):
        scalar = 2.0
        result_matrix = self.sci_matrix * scalar
        self.assertIsInstance(result_matrix, SciMatrix)
        values = np.array(list(result_matrix.data.values()))
        self.assertTrue(np.allclose(values, self.values * scalar))

        result_matrix = self.sci_matrix / scalar
        self.assertIsInstance(result_matrix, SciMatrix)
        values = np.array(list(result_matrix.data.values()))
        self.assertTrue(np.allclose(values, self.values / scalar))

    def test_inplace_operations(self):
        scalar = 2.0
        other_matrix = SciMatrix(
            self.shape, self.row_indices, self.col_indices, self.values
        )
        self.sci_matrix += other_matrix
        values = np.array(list(self.sci_matrix.data.values()))
        self.assertTrue(np.allclose(values, self.values * 2))

        self.sci_matrix -= other_matrix
        values = np.array(list(self.sci_matrix.data.values()))
        self.assertTrue(np.allclose(values, self.values))

        self.sci_matrix *= scalar
        values = np.array(list(self.sci_matrix.data.values()))
        self.assertTrue(np.allclose(values, self.values * scalar))

        self.sci_matrix /= scalar
        values = np.array(list(self.sci_matrix.data.values()))
        self.assertTrue(np.allclose(values, self.values))

    def test_transpose(self):
        transposed_matrix = self.sci_matrix.T
        self.assertIsInstance(transposed_matrix, SciMatrix)

        values = np.array(list(transposed_matrix.data.values()))
        rows = np.array(transposed_matrix.data.nonzero()[0])
        cols = np.array(transposed_matrix.data.nonzero()[1])
        self.assertTrue(np.allclose(values, self.values))
        self.assertTrue(np.allclose(rows, self.col_indices))
        self.assertTrue(np.allclose(cols, self.row_indices))

    def test_inverse(self):
        inv_matrix = self.sci_matrix.inv
        self.assertIsInstance(inv_matrix, SciMatrix)
        dense_product = inv_matrix.to_dense() @ self.sci_matrix.to_dense()
        identity_matrix = np.eye(self.shape[0])
        self.assertTrue(np.allclose(dense_product, identity_matrix))

    def test_determinant(self):
        det = self.sci_matrix.det
        self.assertIsInstance(det, list)
        self.assertEqual(len(det), 1)
        self.assertAlmostEqual(det[0], np.linalg.det(self.sci_matrix.data.toarray()))

    def peformance_test(self):
        import time

        matrix_size = 10_000_000

        # init
        start_time = time.time()
        matrix = SciMatrix.identity((matrix_size, matrix_size))
        end_time = time.time()
        print(f"Init time: {end_time - start_time} s")

        # add
        start_time = time.time()
        mat2 = matrix + matrix
        end_time = time.time()
        print(f"Add time: {end_time - start_time} s")

        # sub
        start_time = time.time()
        mat3 = mat2 - matrix
        end_time = time.time()
        print(f"Sub time: {end_time - start_time} s")

        # mul
        start_time = time.time()
        matrix *= 3.0
        end_time = time.time()
        print(f"Mul time: {end_time - start_time} s")


class TestSparseMatrix(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        print(f"\n---------- Testing < {cls.__name__} > \n")

    @classmethod
    def tearDownClass(cls):
        print("\n---------- Done \n")

    def setUp(self):
        self.dtype = torch.float64
        if settings.FPTYPE == "fp16":
            self.dtype = torch.float16
        elif settings.FPTYPE == "fp32":
            self.dtype = torch.float32

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.shape = (3, 3)
        self.indices = torch.tensor(
            [[0, 1, 2], [0, 1, 2]], dtype=torch.long, device=self.device
        )
        self.values = torch.tensor(
            [1.0, 2.0, 3.0], dtype=torch.float32, device=self.device
        )
        self.sparse_matrix = SparseMatrix(
            self.shape, self.indices, self.values, VariableType.SCALAR, self.device
        )

    def test_initialization(self):
        self.assertEqual(self.sparse_matrix.shape, self.shape)
        self.assertEqual(self.sparse_matrix.dtype, VariableType.SCALAR)
        self.assertEqual(len(self.sparse_matrix.data), 1)

    def test_from_data(self):
        matrix = SparseMatrix.from_data(
            self.shape, self.indices, self.values, self.device
        )
        self.assertEqual(matrix.shape, self.shape)
        self.assertEqual(matrix.dtype, VariableType.SCALAR)
        self.assertEqual(len(matrix.data), 1)

    def test_identity(self):
        identity_matrix = SparseMatrix.identity(
            self.shape, VariableType.SCALAR, self.device
        )
        self.assertEqual(identity_matrix.shape, self.shape)
        values = torch.tensor(
            identity_matrix.data[0].to_dense(), dtype=self.dtype, device=self.device
        )
        self.assertTrue(
            torch.allclose(
                values,
                torch.eye(self.shape[0], device=self.device, dtype=self.dtype),
            )
        )

    def test_zeros(self):
        zeros_matrix = SparseMatrix.zeros(self.shape, VariableType.SCALAR, self.device)
        self.assertEqual(zeros_matrix.shape, self.shape)
        values = torch.tensor(
            zeros_matrix.data[0].to_dense(), dtype=self.dtype, device=self.device
        )
        self.assertTrue(
            torch.allclose(
                values,
                torch.zeros(
                    self.shape,
                    device=self.device,
                    dtype=self.dtype,
                ),
            )
        )

    def test_getitem(self):
        self.assertEqual(self.sparse_matrix[(0, 0)][0], 1.0)
        self.assertEqual(self.sparse_matrix[(1, 1)][0], 2.0)
        self.assertEqual(self.sparse_matrix[(2, 2)][0], 3.0)
        self.assertEqual(self.sparse_matrix[(0, 1)][0], 0.0)  # 非零元素位置应返回0

    def test_setitem(self):
        self.sparse_matrix[(0, 1)] = 4.0
        self.assertEqual(self.sparse_matrix[(0, 1)][0], 4.0)

    def test_add(self):
        other_matrix = SparseMatrix(
            self.shape, self.indices, self.values, VariableType.SCALAR, self.device
        )
        result_matrix = self.sparse_matrix + other_matrix
        expected_values = torch.tensor(
            [2.0, 4.0, 6.0], device=self.device, dtype=self.dtype
        )
        self.assertEqual(result_matrix.shape, self.shape)
        values = torch.tensor(
            list(result_matrix.data[0].data.values()),
            dtype=self.dtype,
            device=self.device,
        )
        self.assertTrue(torch.allclose(values, expected_values))

    def test_iadd(self):
        other_matrix = SparseMatrix(
            self.shape, self.indices, self.values, VariableType.SCALAR, self.device
        )
        self.sparse_matrix += other_matrix
        expected_values = torch.tensor(
            [2.0, 4.0, 6.0], device=self.device, dtype=self.dtype
        )
        values = torch.tensor(
            list(self.sparse_matrix.data[0].data.values()),
            dtype=self.dtype,
            device=self.device,
        )
        self.assertTrue(torch.allclose(values, expected_values))

    def test_sub(self):
        other_matrix = SparseMatrix(
            self.shape, self.indices, self.values, VariableType.SCALAR, self.device
        )
        result_matrix = self.sparse_matrix - other_matrix
        expected_values = torch.tensor(
            [0.0, 0.0, 0.0], device=self.device, dtype=self.dtype
        )
        values = torch.tensor(
            list(result_matrix.data[0].data.values()),
            dtype=self.dtype,
            device=self.device,
        )
        self.assertTrue(len(values) == 0 or torch.allclose(values, expected_values))

    def test_isub(self):
        other_matrix = SparseMatrix(
            self.shape, self.indices, self.values, VariableType.SCALAR, self.device
        )
        self.sparse_matrix -= other_matrix
        expected_values = torch.tensor(
            [0.0, 0.0, 0.0], device=self.device, dtype=self.dtype
        )
        values = torch.tensor(
            list(self.sparse_matrix.data[0].data.values()),
            dtype=self.dtype,
            device=self.device,
        )
        self.assertTrue(len(values) == 0 or torch.allclose(values, expected_values))

    def test_mul(self):
        result_matrix = self.sparse_matrix * 2.0
        expected_values = torch.tensor(
            [2.0, 4.0, 6.0], device=self.device, dtype=self.dtype
        )
        values = torch.tensor(
            list(result_matrix.data[0].data.values()),
            dtype=self.dtype,
            device=self.device,
        )
        self.assertTrue(torch.allclose(values, expected_values))

    def test_imul(self):
        self.sparse_matrix *= 2.0
        expected_values = torch.tensor(
            [2.0, 4.0, 6.0], device=self.device, dtype=self.dtype
        )
        values = torch.tensor(
            list(self.sparse_matrix.data[0].data.values()),
            dtype=self.dtype,
            device=self.device,
        )
        self.assertTrue(torch.allclose(values, expected_values))

    def test_truediv(self):
        result_matrix = self.sparse_matrix / 2.0
        expected_values = torch.tensor(
            [0.5, 1.0, 1.5], device=self.device, dtype=self.dtype
        )
        values = torch.tensor(
            list(result_matrix.data[0].data.values()),
            dtype=self.dtype,
            device=self.device,
        )
        self.assertTrue(torch.allclose(values, expected_values))

    def test_itruediv(self):
        self.sparse_matrix /= 2.0
        expected_values = torch.tensor(
            [0.5, 1.0, 1.5], device=self.device, dtype=self.dtype
        )
        values = torch.tensor(
            list(self.sparse_matrix.data[0].data.values()),
            dtype=self.dtype,
            device=self.device,
        )
        self.assertTrue(torch.allclose(values, expected_values))

    def test_neg(self):
        result_matrix = -self.sparse_matrix
        expected_values = torch.tensor(
            [-1.0, -2.0, -3.0], dtype=self.dtype, device=self.device
        )
        values = torch.tensor(
            list(result_matrix.data[0].data.values()),
            dtype=self.dtype,
            device=self.device,
        )
        self.assertTrue(torch.allclose(values, expected_values))

    def test_abs(self):
        self.sparse_matrix[(0, 1)] = -4.0
        result_matrix = abs(self.sparse_matrix)
        expected_values = torch.tensor(
            [1.0, 4.0, 2.0, 3.0], dtype=self.dtype, device=self.device
        )
        values = torch.tensor(
            list(result_matrix.data[0].data.values()),
            dtype=self.dtype,
            device=self.device,
        )
        self.assertTrue(torch.allclose(values, expected_values))

    def test_to_dense(self):
        dense_matrix = self.sparse_matrix.to_dense()
        expected_dense = np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]])
        self.assertTrue(np.allclose(dense_matrix, expected_dense.reshape(3, 3, 1)))

    def test_T(self):
        transposed_matrix = self.sparse_matrix.T
        self.assertEqual(transposed_matrix.shape, (self.shape[1], self.shape[0]))
        values0 = torch.tensor(list(transposed_matrix.data[0].data.values()))
        values1 = torch.tensor(list(self.sparse_matrix.data[0].data.values()))
        self.assertTrue(
            torch.allclose(
                values0,
                values1,
            )
        )

    def test_inv(self):
        identity_matrix = SparseMatrix.identity(
            (3, 3), VariableType.SCALAR, self.device
        )
        inv_matrix = identity_matrix.inv
        values = torch.tensor(
            inv_matrix.data[0].to_dense(), dtype=self.dtype, device=self.device
        )
        self.assertTrue(
            torch.allclose(
                values,
                torch.eye(3, device=self.device, dtype=self.dtype),
            )
        )

    def test_det(self):
        identity_matrix = SparseMatrix.identity(
            (3, 3), VariableType.SCALAR, self.device
        )
        self.assertAlmostEqual(identity_matrix.det[0], 1.0)


if __name__ == "__main__":
    with open("./tests/reports/report.txt", "w", encoding="utf8") as reporter:
        runner = unittest.TextTestRunner(stream=reporter, verbosity=2)

        suit = unittest.TestLoader().loadTestsFromTestCase(TestTorchMatrix)
        runner.run(suit)

        # suit = unittest.TestLoader().loadTestsFromTestCase(TestCupyMatrix)
        # runner.run(suit)

        # suit = unittest.TestLoader().loadTestsFromTestCase(TestSciMatrix)
        # runner.run(suit)

        # suit = unittest.TestLoader().loadTestsFromTestCase(TestSparseMatrix)
        # runner.run(suit)
