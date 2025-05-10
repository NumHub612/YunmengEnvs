# -*- encoding: utf-8 -*-
import unittest
import torch
import numpy as np
from configs.settings import settings
from core.numerics.fields import VariableType
from core.numerics.matrix import TorchMatrix


class TestTorchMatrix(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        print(f"\n---------- Testing < {cls.__name__} > \n")

    @classmethod
    def tearDownClass(cls):
        print("\n---------- Done \n")

    def setUp(self):
        self.device = settings.DEVICE
        self.shape = (3, 3)
        self.indices = torch.tensor(
            [[0, 1, 2], [0, 1, 2]], dtype=torch.long, device=self.device
        )
        self.values = torch.tensor(
            [1.0, 2.0, 3.0], dtype=settings.DTYPE, device=self.device
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
        expected_values = torch.ones(3, dtype=settings.DTYPE, device=self.device)
        self.assertTrue(torch.equal(identity_matrix.data.indices(), expected_indices))
        self.assertTrue(torch.equal(identity_matrix.data.values(), expected_values))

    def test_ones(self):
        ones_matrix = TorchMatrix.ones((2, 2))
        expected_indices = torch.tensor(
            [[0, 0, 1, 1], [0, 1, 0, 1]], dtype=torch.long, device=self.device
        )
        expected_values = torch.ones(4, dtype=settings.DTYPE, device=self.device)
        self.assertTrue(torch.equal(ones_matrix.data.indices(), expected_indices))
        self.assertTrue(torch.equal(ones_matrix.data.values(), expected_values))

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
            [2.0, 4.0, 6.0], dtype=settings.DTYPE, device=self.device
        )
        self.assertTrue(torch.equal(result_matrix.data.values(), expected_values))

    def test_sub(self):
        other_matrix = TorchMatrix(self.shape, self.indices, self.values)
        result_matrix = self.matrix - other_matrix
        expected_values = torch.tensor(
            [0.0, 0.0, 0.0], dtype=settings.DTYPE, device=self.device
        )
        self.assertTrue(torch.equal(result_matrix.data.values(), expected_values))

    def test_mul(self):
        other_matrix = TorchMatrix(self.shape, self.indices, self.values)
        result_matrix = self.matrix * other_matrix
        expected_values = torch.tensor(
            [1.0, 4.0, 9.0], dtype=settings.DTYPE, device=self.device
        )
        self.assertTrue(torch.equal(result_matrix.data.values(), expected_values))

    def test_truediv(self):
        result_matrix = self.matrix / 2.0
        expected_values = torch.tensor(
            [0.5, 1.0, 1.5], dtype=settings.DTYPE, device=self.device
        )
        self.assertTrue(torch.equal(result_matrix.data.values(), expected_values))

    def test_neg(self):
        neg_matrix = -self.matrix
        expected_values = torch.tensor(
            [-1.0, -2.0, -3.0], dtype=settings.DTYPE, device=self.device
        )
        self.assertTrue(torch.equal(neg_matrix.data.values(), expected_values))

    def test_abs(self):
        abs_matrix = abs(self.matrix)
        expected_values = torch.tensor(
            [1.0, 2.0, 3.0], dtype=settings.DTYPE, device=self.device
        )
        self.assertTrue(torch.equal(abs_matrix.data.values(), expected_values))

    def test_peformance(self):
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


if __name__ == "__main__":
    with open("./tests/reports/report.txt", "w", encoding="utf8") as reporter:
        suit = unittest.TestLoader().loadTestsFromTestCase(TestTorchMatrix)

        runner = unittest.TextTestRunner(stream=reporter, verbosity=2)
        runner.run(suit)
