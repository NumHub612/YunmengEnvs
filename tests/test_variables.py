# -*- encoding: utf-8 -*-
import unittest
import numpy as np
from core.numerics.fields import Scalar, Vector, Tensor, VariableType
from configs.settings import settings


class TestVariables(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        print(f"\n---------- Testing < {cls.__name__} > \n")

    @classmethod
    def tearDownClass(cls):
        print("\n---------- Done \n")

    def setUp(self):
        self.scalar = Scalar(5.0)
        self.vector = Vector(1.0, 2.0, 3.0)
        self.tensor = Tensor(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0)

    def test_from_torch(self):
        scalar_torch = np.array([5.0], dtype=np.float64)
        vector_torch = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        tensor_torch = np.array(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], dtype=np.float64
        )

        self.assertEqual(Scalar.from_data(scalar_torch).value, self.scalar.value)
        self.assertTrue(
            np.allclose(Vector.from_data(vector_torch)._value, self.vector._value)
        )
        self.assertTrue(
            np.allclose(Tensor.from_data(tensor_torch)._value, self.tensor._value)
        )

    def test_from_np(self):
        scalar_np = np.array([5.0])
        vector_np = np.array([1.0, 2.0, 3.0])
        tensor_np = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])

        self.assertEqual(Scalar.from_data(scalar_np).value, self.scalar.value)
        self.assertTrue(
            np.allclose(Vector.from_data(vector_np).to_np(), self.vector.to_np())
        )
        self.assertTrue(
            np.allclose(Tensor.from_data(tensor_np).to_np(), self.tensor.to_np())
        )

    def test_to_np(self):
        self.assertTrue(np.allclose(self.scalar.to_np(), np.array([5.0])))
        self.assertTrue(np.allclose(self.vector.to_np(), np.array([1.0, 2.0, 3.0])))
        self.assertTrue(
            np.allclose(
                self.tensor.to_np(),
                np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]),
            )
        )

    def test_magnitude(self):
        self.assertEqual(self.scalar.magnitude, abs(5.0))
        self.assertAlmostEqual(self.vector.magnitude, np.linalg.norm([1.0, 2.0, 3.0]))
        self.assertAlmostEqual(
            self.tensor.magnitude, np.linalg.norm(self.tensor.to_np())
        )

    def test_data(self):
        self.assertTrue(
            np.allclose(self.scalar.data, np.array([5.0], dtype=np.float64))
        )
        self.assertTrue(
            np.allclose(self.vector.data, np.array([1.0, 2.0, 3.0], dtype=np.float64))
        )
        self.assertTrue(
            np.allclose(
                self.tensor.data,
                np.array(
                    [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
                    dtype=np.float64,
                ),
            )
        )

    def test_type(self):
        self.assertEqual(self.scalar.type, VariableType.SCALAR)
        self.assertEqual(self.vector.type, VariableType.VECTOR)
        self.assertEqual(self.tensor.type, VariableType.TENSOR)

    def test_unit(self):
        self.assertEqual(Scalar.unit().value, 1.0)
        self.assertTrue(
            np.allclose(
                Vector.unit()._value,
                np.array([1.0, 1.0, 1.0], dtype=np.float64),
            )
        )
        self.assertTrue(
            np.allclose(
                Tensor.unit()._value,
                np.array(
                    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                    dtype=np.float64,
                ),
            )
        )

    def test_zero(self):
        self.assertEqual(Scalar.zero().value, 0.0)
        self.assertTrue(
            np.allclose(
                Vector.zero()._value,
                np.array([0.0, 0.0, 0.0], dtype=np.float64),
            )
        )
        self.assertTrue(
            np.allclose(
                Tensor.zero()._value,
                np.array(
                    [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                    dtype=np.float64,
                ),
            )
        )

    def test_arithmetic_operations(self):
        scalar1 = Scalar(5.0)
        scalar2 = Scalar(3.0)
        vector1 = Vector(1.0, 2.0, 3.0)
        vector2 = Vector(4.0, 5.0, 6.0)
        tensor1 = Tensor(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0)
        tensor2 = Tensor(9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0)

        # add
        self.assertEqual((scalar1 + scalar2).value, 8.0)
        self.assertTrue(
            np.allclose(
                (vector1 + vector2)._value,
                np.array([5.0, 7.0, 9.0], dtype=np.float64),
            )
        )
        self.assertTrue(
            np.allclose(
                (tensor1 + tensor2)._value,
                np.array(
                    [[10.0, 10.0, 10.0], [10.0, 10.0, 10.0], [10.0, 10.0, 10.0]],
                    dtype=np.float64,
                ),
            )
        )

        # sub
        self.assertEqual((scalar1 - scalar2).value, 2.0)
        self.assertTrue(
            np.allclose(
                (vector1 - vector2)._value,
                np.array([-3.0, -3.0, -3.0], dtype=np.float64),
            )
        )
        self.assertTrue(
            np.allclose(
                (tensor1 - tensor2)._value,
                np.array(
                    [[-8.0, -6.0, -4.0], [-2.0, 0.0, 2.0], [4.0, 6.0, 8.0]],
                    dtype=np.float64,
                ),
            )
        )

        # mul
        self.assertEqual((scalar1 * scalar2).value, 15.0)
        self.assertTrue(
            np.allclose(
                (vector1 * scalar2)._value,
                np.array([3.0, 6.0, 9.0], dtype=np.float64),
            )
        )
        self.assertTrue(
            np.allclose(
                (tensor1 * scalar2)._value,
                np.array(
                    [[3.0, 6.0, 9.0], [12.0, 15.0, 18.0], [21.0, 24.0, 27.0]],
                    dtype=np.float64,
                ),
            )
        )

        # div
        self.assertEqual((scalar1 / scalar2).value, 5.0 / 3.0)
        self.assertTrue(
            np.allclose(
                (vector1 / scalar2)._value,
                np.array([1.0 / 3.0, 2.0 / 3.0, 1.0], dtype=np.float64),
            )
        )
        self.assertTrue(
            np.allclose(
                (tensor1 / scalar2)._value,
                np.array(
                    [
                        [1.0 / 3.0, 2.0 / 3.0, 1.0],
                        [4.0 / 3.0, 5.0 / 3.0, 2.0],
                        [7.0 / 3.0, 8.0 / 3.0, 3.0],
                    ],
                    dtype=np.float64,
                ),
            )
        )

    def test_comparison_operations(self):
        scalar1 = Scalar(5.0)
        scalar2 = Scalar(5.0)
        vector1 = Vector(1.0, 2.0, 3.0)
        vector2 = Vector(1.0, 2.0, 3.0)
        tensor1 = Tensor(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0)
        tensor2 = Tensor(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0)

        self.assertTrue(scalar1 == scalar2)
        self.assertTrue(vector1 == vector2)
        self.assertTrue(tensor1 == tensor2)

        self.assertFalse(scalar1 != scalar2)
        self.assertFalse(vector1 != vector2)
        self.assertFalse(tensor1 != tensor2)


if __name__ == "__main__":
    with open("./tests/reports/report.txt", "w", encoding="utf8") as reporter:
        suit = unittest.TestLoader().loadTestsFromTestCase(TestVariables)

        runner = unittest.TextTestRunner(stream=reporter, verbosity=2)
        runner.run(suit)
