# -*- encoding: utf-8 -*-
import unittest
import torch
import numpy as np
from core.numerics.fields import (
    Field,
    Scalar,
    Vector,
    Tensor,
    ElementType,
    VariableType,
)


class TestFields(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        print(f"\n---------- Testing < {cls.__name__} > \n")

    @classmethod
    def tearDownClass(cls):
        print("\n---------- Done \n")

    def setUp(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.scalar_field = Field(
            3, ElementType.NODE, VariableType.SCALAR, Scalar(1.0), device=self.device
        )
        self.vector_field = Field(
            3,
            ElementType.NODE,
            VariableType.VECTOR,
            Vector(1.0, 2.0, 3.0),
            device=self.device,
        )
        self.tensor_field = Field(
            3,
            ElementType.NODE,
            VariableType.TENSOR,
            Tensor(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0),
            device=self.device,
        )

    def test_initialization(self):
        self.assertEqual(self.scalar_field.size, 3)
        self.assertEqual(self.scalar_field.dtype, VariableType.SCALAR)
        self.assertEqual(self.scalar_field.etype, ElementType.NODE)
        self.assertTrue(
            torch.allclose(
                self.scalar_field.data,
                torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64, device=self.device),
            )
        )

        self.assertEqual(self.vector_field.size, 3)
        self.assertEqual(self.vector_field.dtype, VariableType.VECTOR)
        self.assertEqual(self.vector_field.etype, ElementType.NODE)
        self.assertTrue(
            torch.allclose(
                self.vector_field.data,
                torch.tensor(
                    [[1.0, 2.0, 3.0]] * 3, dtype=torch.float64, device=self.device
                ),
            )
        )

        self.assertEqual(self.tensor_field.size, 3)
        self.assertEqual(self.tensor_field.dtype, VariableType.TENSOR)
        self.assertEqual(self.tensor_field.etype, ElementType.NODE)
        self.assertTrue(
            torch.allclose(
                self.tensor_field.data,
                torch.tensor(
                    [[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]] * 3,
                    dtype=torch.float64,
                    device=self.device,
                ),
            )
        )

    def test_from_torch(self):
        scalar_torch = torch.tensor(
            [1.0, 2.0, 3.0], dtype=torch.float64, device=self.device
        )
        vector_torch = torch.tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
            dtype=torch.float64,
            device=self.device,
        )
        tensor_torch = torch.tensor(
            [[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]] * 3,
            dtype=torch.float64,
            device=self.device,
        )

        scalar_field = Field.from_torch(
            scalar_torch, ElementType.NODE, "scalar", self.device
        )
        vector_field = Field.from_torch(
            vector_torch, ElementType.NODE, "vector", self.device
        )
        tensor_field = Field.from_torch(
            tensor_torch, ElementType.NODE, "tensor", self.device
        )

        self.assertTrue(torch.allclose(scalar_field.data, scalar_torch))
        self.assertTrue(torch.allclose(vector_field.data, vector_torch))
        self.assertTrue(torch.allclose(tensor_field.data, tensor_torch))

    def test_to_np(self):
        self.assertTrue(
            np.allclose(self.scalar_field.to_np(), np.array([1.0, 1.0, 1.0]))
        )
        self.assertTrue(
            np.allclose(self.vector_field.to_np(), np.array([[1.0, 2.0, 3.0]] * 3))
        )
        self.assertTrue(
            np.allclose(
                self.tensor_field.to_np(),
                np.array([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]] * 3),
            )
        )

    def test_scalarize(self):
        scalar_fields = self.vector_field.scalarize()
        self.assertEqual(len(scalar_fields), 3)
        self.assertTrue(
            torch.allclose(
                scalar_fields[0].data,
                torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64, device=self.device),
            )
        )
        self.assertTrue(
            torch.allclose(
                scalar_fields[1].data,
                torch.tensor([2.0, 2.0, 2.0], dtype=torch.float64, device=self.device),
            )
        )
        self.assertTrue(
            torch.allclose(
                scalar_fields[2].data,
                torch.tensor([3.0, 3.0, 3.0], dtype=torch.float64, device=self.device),
            )
        )

    def test_assign(self):
        new_scalar_field = Field(
            3, ElementType.NODE, VariableType.SCALAR, Scalar(5.0), device=self.device
        )
        self.scalar_field.assign(new_scalar_field)
        self.assertTrue(
            torch.allclose(
                self.scalar_field.data,
                torch.tensor([5.0, 5.0, 5.0], dtype=torch.float64, device=self.device),
            )
        )

        new_vector = Vector(10.0, 20.0, 30.0)
        self.vector_field.assign(new_vector)
        self.assertTrue(
            torch.allclose(
                self.vector_field.data,
                torch.tensor(
                    [[10.0, 20.0, 30.0]] * 3, dtype=torch.float64, device=self.device
                ),
            )
        )

    def test_arithmetic_operations(self):
        scalar_field1 = Field(
            3, ElementType.NODE, VariableType.SCALAR, Scalar(1.0), device=self.device
        )
        scalar_field2 = Field(
            3, ElementType.NODE, VariableType.SCALAR, Scalar(2.0), device=self.device
        )

        vector_field1 = Field(
            3,
            ElementType.NODE,
            VariableType.VECTOR,
            Vector(1.0, 2.0, 3.0),
            device=self.device,
        )
        vector_field2 = Field(
            3,
            ElementType.NODE,
            VariableType.VECTOR,
            Vector(4.0, 5.0, 6.0),
            device=self.device,
        )

        # add
        self.assertTrue(
            torch.allclose(
                (scalar_field1 + scalar_field2).data,
                torch.tensor([3.0, 3.0, 3.0], dtype=torch.float64, device=self.device),
            )
        )
        self.assertTrue(
            torch.allclose(
                (vector_field1 + vector_field2).data,
                torch.tensor(
                    [[5.0, 7.0, 9.0]] * 3, dtype=torch.float64, device=self.device
                ),
            )
        )

        # sub
        self.assertTrue(
            torch.allclose(
                (scalar_field1 - scalar_field2).data,
                torch.tensor(
                    [-1.0, -1.0, -1.0], dtype=torch.float64, device=self.device
                ),
            )
        )
        self.assertTrue(
            torch.allclose(
                (vector_field1 - vector_field2).data,
                torch.tensor(
                    [[-3.0, -3.0, -3.0]] * 3, dtype=torch.float64, device=self.device
                ),
            )
        )

        # mul
        self.assertTrue(
            torch.allclose(
                (scalar_field1 * 2.0).data,
                torch.tensor([2.0, 2.0, 2.0], dtype=torch.float64, device=self.device),
            )
        )
        self.assertTrue(
            torch.allclose(
                (vector_field1 * 2.0).data,
                torch.tensor(
                    [[2.0, 4.0, 6.0]] * 3, dtype=torch.float64, device=self.device
                ),
            )
        )

        # div
        self.assertTrue(
            torch.allclose(
                (scalar_field1 / 2.0).data,
                torch.tensor([0.5, 0.5, 0.5], dtype=torch.float64, device=self.device),
            )
        )
        self.assertTrue(
            torch.allclose(
                (vector_field1 / 2.0).data,
                torch.tensor(
                    [[0.5, 1.0, 1.5]] * 3, dtype=torch.float64, device=self.device
                ),
            )
        )


if __name__ == "__main__":
    with open("./tests/reports/report.txt", "w", encoding="utf8") as reporter:
        suit = unittest.TestLoader().loadTestsFromTestCase(TestFields)

        runner = unittest.TextTestRunner(stream=reporter, verbosity=2)
        runner.run(suit)
