# -*- encoding: utf-8 -*-
import unittest
import numpy as np
import torch
import time
from core.numerics.mats import LinearEqs, SparseMatrix
from core.numerics.fields import Field, ElementType, VariableType, Scalar
from configs.settings import settings


class TestLinearEqs(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        print(f"\n---------- Testing < {cls.__name__} > \n")

    @classmethod
    def tearDownClass(cls):
        print("\n---------- Done \n")

    def setUp(self):
        self.device = torch.device(settings.DEVICE)
        self.dtype = np.float64
        if settings.FPTYPE == "fp16":
            self.dtype = np.float16
        elif settings.FPTYPE == "fp32":
            self.dtype = np.float32

    def test_initialization(self):
        mat = SparseMatrix.from_data(
            (3, 3),
            indices=np.array([[0, 1, 2], [1, 2, 0]]),
            values=np.array([1.0, 2.0, 3.0]),
            device=self.device,
        )
        rhs = Field(
            size=3,
            element_type=ElementType.CELL,
            data_type=VariableType.SCALAR,
            device=self.device,
        )
        eqs = LinearEqs("x", mat, rhs, self.device)

        self.assertEqual(eqs.variable, "x")
        self.assertEqual(eqs.matrix.shape, (3, 3))
        self.assertEqual(eqs.rhs.size, 3)

    def test_matrix_and_rhs_setters(self):
        mat = SparseMatrix.from_data(
            (3, 3),
            indices=np.array([[0, 1, 2], [1, 2, 0]]),
            values=np.array([1.0, 2.0, 3.0]),
            device=self.device,
        )
        rhs = Field(
            size=3,
            element_type=ElementType.CELL,
            data_type=VariableType.SCALAR,
            device=self.device,
        )
        eqs = LinearEqs("x", mat, rhs, self.device)

        new_mat = SparseMatrix.from_data(
            (3, 3),
            indices=np.array([[0, 1, 2], [1, 2, 0]]),
            values=np.array([4.0, 5.0, 6.0]),
            device=self.device,
        )
        eqs.matrix = new_mat
        self.assertEqual(eqs.matrix, new_mat)

        new_rhs = Field(
            size=3,
            element_type=ElementType.CELL,
            data_type=VariableType.SCALAR,
            device=self.device,
        )
        eqs.rhs = new_rhs
        self.assertEqual(eqs.rhs, new_rhs)

    def test_addition(self):
        mat1 = SparseMatrix.from_data(
            (3, 3),
            indices=np.array([[0, 1, 2], [1, 2, 0]]),
            values=np.array([1.0, 2.0, 3.0]),
            device=self.device,
        )
        mat2 = SparseMatrix.from_data(
            (3, 3),
            indices=np.array([[0, 1, 2], [1, 2, 0]]),
            values=np.array([4.0, 5.0, 6.0]),
            device=self.device,
        )
        rhs1 = Field(
            size=3,
            element_type=ElementType.CELL,
            data_type=VariableType.SCALAR,
            device=self.device,
        )
        rhs2 = Field(
            size=3,
            element_type=ElementType.CELL,
            data_type=VariableType.SCALAR,
            device=self.device,
        )
        eqs1 = LinearEqs("x", mat1, rhs1, self.device)
        eqs2 = LinearEqs("x", mat2, rhs2, self.device)

        result = eqs1 + eqs2
        self.assertEqual(result.matrix[1, 2], [7.0])
        self.assertEqual(result.rhs[0], 0.0)

    def test_subtraction(self):
        mat1 = SparseMatrix.from_data(
            (3, 3),
            indices=np.array([[0, 1, 2], [1, 2, 0]]),
            values=np.array([1.0, 2.0, 3.0]),
            device=self.device,
        )
        mat2 = SparseMatrix.from_data(
            (3, 3),
            indices=np.array([[0, 1, 2], [1, 2, 0]]),
            values=np.array([4.0, 5.0, 6.0]),
            device=self.device,
        )
        rhs1 = Field(
            size=3,
            element_type=ElementType.CELL,
            data_type=VariableType.SCALAR,
            data=Scalar(1.0),
            device=self.device,
        )
        rhs2 = Field(
            size=3,
            element_type=ElementType.CELL,
            data_type=VariableType.SCALAR,
            data=Scalar(2.0),
            device=self.device,
        )
        eqs1 = LinearEqs("x", mat1, rhs1, self.device)
        eqs2 = LinearEqs("x", mat2, rhs2, self.device)

        result = eqs1 - eqs2
        self.assertEqual(result.matrix[1, 2], (mat1 - mat2)[1, 2])
        self.assertEqual(result.rhs[1], -1.0)

    def test_negation(self):
        mat = SparseMatrix.from_data(
            (3, 3),
            indices=np.array([[0, 1, 2], [1, 2, 0]]),
            values=np.array([1.0, 2.0, 3.0]),
            device=self.device,
        )
        rhs = Field(
            size=3,
            element_type=ElementType.CELL,
            data_type=VariableType.SCALAR,
            data=Scalar(1.0),
            device=self.device,
        )
        eqs = LinearEqs("x", mat, rhs, self.device)

        result = -eqs
        self.assertEqual(result.matrix[1, 2], [-2.0])
        self.assertEqual(result.rhs[1], -1.0)

    def test_solve(self):
        mat = SparseMatrix.from_data(
            (3, 3),
            indices=np.array([[0, 1, 2], [1, 2, 0]]),
            values=np.array([1.0, 2.0, 3.0]),
            device=self.device,
        )
        rhs = Field(
            size=3,
            element_type=ElementType.CELL,
            data_type=VariableType.SCALAR,
            device=self.device,
        )

        eqs = LinearEqs("x", mat, rhs, self.device)
        solution = eqs.solve()

        self.assertIsInstance(solution, Field)
        self.assertEqual(solution.size, 3)

    def test_solve_with_zero_matrix(self):
        mat = SparseMatrix.zeros((3, 3), device=self.device)
        rhs = Field(
            size=3,
            element_type=ElementType.CELL,
            data_type=VariableType.SCALAR,
            device=self.device,
        )
        eqs = LinearEqs("x", mat, rhs, self.device)

        solution = eqs.solve()
        self.assertIsInstance(solution, Field)
        self.assertEqual(solution.size, 3)
        np.testing.assert_array_equal(solution.to_np(), rhs.to_np())

    def test_incompatible_shapes(self):
        mat = SparseMatrix.from_data(
            (3, 3),
            indices=np.array([[0, 1, 2], [1, 2, 0]]),
            values=np.array([1.0, 2.0, 3.0]),
            device=self.device,
        )
        rhs = Field(
            size=4,
            element_type=ElementType.CELL,
            data_type=VariableType.SCALAR,
            device=self.device,
        )
        with self.assertRaises(ValueError):
            LinearEqs("x", mat, rhs, self.device)

    def test_incompatible_types(self):
        mat = SparseMatrix.from_data(
            (3, 3),
            indices=np.array([[0, 1, 2], [1, 2, 0]]),
            values=np.array([1.0, 2.0, 3.0]),
            device=self.device,
        )
        rhs = Field(
            size=4,
            element_type=ElementType.CELL,
            data_type=VariableType.SCALAR,
            device=self.device,
        )
        with self.assertRaises(ValueError):
            LinearEqs("x", mat, rhs, self.device)

    def test_scalarize(self):
        mat = SparseMatrix.from_data(
            (3, 3),
            indices=np.array([[0, 1, 2], [1, 2, 0]]),
            values=np.array([1.0, 2.0, 3.0]),
            device=self.device,
        )
        rhs = Field(
            size=3,
            element_type=ElementType.CELL,
            data_type=VariableType.SCALAR,
            device=self.device,
        )
        eqs = LinearEqs("x", mat, rhs, self.device)

        scalarized_eqs = eqs.scalarize()
        self.assertEqual(len(scalarized_eqs), 1)

    def performance_test(self):
        """Test the performance of solving large linear equations."""

        # Test with 1 million equations
        print("Testing with 1 million equations...")
        shape = (1_000_000, 1_000_000)
        mat = SparseMatrix.identity(shape, device=self.device)
        rhs = Field(
            shape[0],
            ElementType.CELL,
            VariableType.SCALAR,
            Scalar(1.0),
            device=self.device,
        )
        eqs = LinearEqs("x", mat, rhs, self.device)

        # Solve using cupy
        start_time = time.time()
        solution_torch = eqs.solve(method="cupy")
        torch_time = time.time() - start_time

        self.assertIsInstance(solution_torch, Field)
        self.assertEqual(solution_torch.size, shape[0])
        np.testing.assert_array_almost_equal(
            solution_torch.to_np().flatten(),
            np.ones(shape[0], dtype=self.dtype),
        )
        print(f"Time taken by cupy: {torch_time:.2f} seconds")

        # solve using scipy
        start_time = time.time()
        solution_scipy = eqs.solve(method="scipy")
        scipy_time = time.time() - start_time

        self.assertIsInstance(solution_scipy, Field)
        self.assertEqual(solution_scipy.size, shape[0])
        np.testing.assert_array_almost_equal(
            solution_scipy.to_np().flatten(),
            np.ones(shape[0]),
        )
        print(f"Time taken by scipy: {scipy_time:.2f} seconds")

        # Test with 10 million equations
        print("\nTesting with 10 million equations...")
        shape = (10_000_000, 10_000_000)
        mat = SparseMatrix.identity(shape, device=self.device)
        rhs = Field(
            shape[0],
            ElementType.CELL,
            VariableType.SCALAR,
            Scalar(2.0),
            device=self.device,
        )
        eqs = LinearEqs("x", mat, rhs, self.device)

        # Solve using cupy
        start_time = time.time()
        solution_torch = eqs.solve(method="cupy")
        torch_time = time.time() - start_time

        self.assertIsInstance(solution_torch, Field)
        self.assertEqual(solution_torch.size, shape[0])
        np.testing.assert_array_almost_equal(
            solution_torch.to_np().flatten(),
            2.0
            * np.ones(
                shape[0],
            ),
        )
        print(f"Time taken by cupy: {torch_time:.2f} seconds")

        # solve using scipy
        start_time = time.time()
        solution_scipy = eqs.solve(method="scipy")
        scipy_time = time.time() - start_time

        self.assertIsInstance(solution_scipy, Field)
        self.assertEqual(solution_scipy.size, shape[0])
        np.testing.assert_array_almost_equal(
            solution_scipy.to_np().flatten(),
            2.0
            * np.ones(
                shape[0],
            ),
        )
        print(f"Time taken by scipy: {scipy_time:.2f} seconds")


if __name__ == "__main__":
    with open("./tests/reports/report.txt", "w", encoding="utf8") as reporter:
        suit = unittest.TestLoader().loadTestsFromTestCase(TestLinearEqs)

        runner = unittest.TextTestRunner(stream=reporter, verbosity=2)
        runner.run(suit)
