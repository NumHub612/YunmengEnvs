# -*- encoding: utf-8 -*-
import unittest
import torch
import time
import numpy as np
import random
from configs.settings import settings
from core.numerics.fields import Field, ElementType, VariableType, DTYPE_MAP


class TestFields(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        print(f"\n---------- Testing < {cls.__name__} > \n")

    @classmethod
    def tearDownClass(cls):
        print("\n---------- Done \n")

    def setUp(self):
        self.device = torch.device(settings.DEVICE)
        self.dtype = torch.float64
        if settings.FPTYPE == "fp16":
            self.dtype = torch.float16
        elif settings.FPTYPE == "fp32":
            self.dtype = torch.float32

        self.size = 100
        self.element_type = ElementType.NODE

    def test_scalar_field(self):
        data = torch.randn(self.size, 1, device=self.device)
        field = Field(
            self.size,
            self.element_type,
            VariableType.SCALAR,
            data,
            device=self.device,
        )
        self.assertEqual(len(field.data), field.chunks)
        self.assertEqual(field.size, self.size)
        self.assertEqual(field.dtype, VariableType.SCALAR)
        self.assertEqual(field.etype, self.element_type)

    def test_vector_field(self):
        data = torch.randn(self.size, 3, device=self.device)
        field = Field(
            self.size,
            self.element_type,
            VariableType.VECTOR,
            data,
            device=self.device,
        )
        self.assertEqual(len(field.data), field.chunks)
        self.assertEqual(field.size, self.size)
        self.assertEqual(field.dtype, VariableType.VECTOR)
        self.assertEqual(field.etype, self.element_type)

    def test_tensor_field(self):
        data = torch.randn(self.size, 3, 3, device=self.device)
        field = Field(
            self.size,
            self.element_type,
            VariableType.TENSOR,
            data,
            device=self.device,
        )
        self.assertEqual(len(field.data), field.chunks)
        self.assertEqual(field.size, self.size)
        self.assertEqual(field.dtype, VariableType.TENSOR)
        self.assertEqual(field.etype, self.element_type)

    def test_addition(self):
        for dtype in [VariableType.SCALAR, VariableType.VECTOR, VariableType.TENSOR]:
            data1 = torch.randn(
                self.size,
                *(
                    (1,)
                    if dtype == VariableType.SCALAR
                    else (3,) if dtype == VariableType.VECTOR else (3, 3)
                ),
                device=self.device,
            )
            data2 = torch.randn(
                self.size,
                *(
                    (1,)
                    if dtype == VariableType.SCALAR
                    else (3,) if dtype == VariableType.VECTOR else (3, 3)
                ),
                device=self.device,
            )
            field1 = Field(
                self.size, self.element_type, dtype, data1, device=self.device
            )
            field2 = Field(
                self.size, self.element_type, dtype, data2, device=self.device
            )
            result = field1 + field2
            expected = data1 + data2
            self.assertTrue(np.allclose(result.to_np(), expected.cpu().numpy()))

    def test_subtraction(self):
        for dtype in [VariableType.SCALAR, VariableType.VECTOR, VariableType.TENSOR]:
            data1 = torch.randn(
                self.size,
                *(
                    (1,)
                    if dtype == VariableType.SCALAR
                    else (3,) if dtype == VariableType.VECTOR else (3, 3)
                ),
                device=self.device,
            )
            data2 = torch.randn(
                self.size,
                *(
                    (1,)
                    if dtype == VariableType.SCALAR
                    else (3,) if dtype == VariableType.VECTOR else (3, 3)
                ),
                device=self.device,
            )
            field1 = Field(
                self.size, self.element_type, dtype, data1, device=self.device
            )
            field2 = Field(
                self.size, self.element_type, dtype, data2, device=self.device
            )
            result = field1 - field2
            expected = data1 - data2
            self.assertTrue(np.allclose(result.to_np(), expected.cpu().numpy()))

    def test_multiplication(self):
        for dtype in [VariableType.SCALAR, VariableType.VECTOR, VariableType.TENSOR]:
            data1 = torch.randn(
                self.size,
                *(
                    (1,)
                    if dtype == VariableType.SCALAR
                    else (3,) if dtype == VariableType.VECTOR else (3, 3)
                ),
                device=self.device,
            )
            scalar = 2.0
            field1 = Field(
                self.size, self.element_type, dtype, data1, device=self.device
            )
            result = field1 * scalar
            expected = data1 * scalar
            self.assertTrue(np.allclose(result.to_np(), expected.cpu().numpy()))

    def test_negation(self):
        for dtype in [VariableType.SCALAR, VariableType.VECTOR, VariableType.TENSOR]:
            data = torch.randn(
                self.size,
                *(
                    (1,)
                    if dtype == VariableType.SCALAR
                    else (3,) if dtype == VariableType.VECTOR else (3, 3)
                ),
                device=self.device,
            )
            field = Field(self.size, self.element_type, dtype, data, device=self.device)
            result = -field
            expected = -data
            self.assertTrue(np.allclose(result.to_np(), expected.cpu().numpy()))

    def test_absolute(self):
        for dtype in [VariableType.SCALAR, VariableType.VECTOR, VariableType.TENSOR]:
            data = torch.randn(
                self.size,
                *(
                    (1,)
                    if dtype == VariableType.SCALAR
                    else (3,) if dtype == VariableType.VECTOR else (3, 3)
                ),
                device=self.device,
            )
            field = Field(self.size, self.element_type, dtype, data, device=self.device)
            result = abs(field)
            expected = torch.abs(data)
            self.assertTrue(np.allclose(result.to_np(), expected.cpu().numpy()))

    def test_indexing(self):
        for dtype in [VariableType.SCALAR, VariableType.VECTOR, VariableType.TENSOR]:
            data = torch.randn(
                self.size,
                *(
                    (1,)
                    if dtype == VariableType.SCALAR
                    else (3,) if dtype == VariableType.VECTOR else (3, 3)
                ),
                device=self.device,
                dtype=torch.float64,
            )
            field1 = Field(
                self.size, self.element_type, dtype, data, device=self.device
            )
            if settings.GPUs:
                default_gpu = [settings.GPUs[0]]
            else:
                default_gpu = []
            field2 = Field(
                self.size,
                self.element_type,
                dtype,
                data,
                device=self.device,
                gpus=default_gpu,
            )
            index = 50
            var = DTYPE_MAP[dtype].from_data(data[index])
            self.assertTrue(np.allclose(field2[index].data, var.data))

    def performance_test(self):
        size = 10_000_000
        element_type = ElementType.NODE
        device = self.device
        gpus = settings.GPUs
        print(f"Testing 1e7 element performance on {device}: {gpus}...")

        for dtype in [VariableType.SCALAR, VariableType.VECTOR, VariableType.TENSOR]:
            print(f"Testing {dtype} type...")
            shape = (
                (size, 1)
                if dtype == VariableType.SCALAR
                else (size, 3) if dtype == VariableType.VECTOR else (size, 3, 3)
            )
            data = torch.randn(*shape)

            # init
            start_time = time.time()
            field = Field(size, element_type, dtype, data, gpus=gpus)
            init_time = time.time() - start_time
            print(f"Initialization time: {init_time:.6f} seconds")

            # quire1
            start_time = time.time()
            for i in range(1000000):
                index = random.randint(0, size - 1)
                field[index]
            quire_time = time.time() - start_time
            print(f"Random quering 1e6 times cost: {quire_time:.6f} seconds")

            # quire2
            start_time = time.time()
            count = 0
            for v in field:
                count += 1
                if count == 1000000:
                    break
            quire_time = time.time() - start_time
            print(f"Sequential quering 1e6 times cost: {quire_time:.6f} seconds")

            # assign1
            start_time = time.time()
            value = data[0]
            for i in range(1000000):
                index = random.randint(1, size - 1)
                field[index] = value
            assign_time = time.time() - start_time
            print(f"Random assigning 1e6 times cost: {assign_time:.6f} seconds")

            # assign2
            start_time = time.time()
            value = data[0]
            for i in range(1000000):
                field[i] = value
            assign_time = time.time() - start_time
            print(f"Sequential assigning 1e6 times cost: {assign_time:.6f} seconds")

            # add
            start_time = time.time()
            result = field + field
            add_time = time.time() - start_time
            print(f"Addition time: {add_time:.6f} seconds")

            # mul
            start_time = time.time()
            result = field * 2.0
            mul_time = time.time() - start_time
            print(f"Multiplication time: {mul_time:.6f} seconds")

            # neg
            start_time = time.time()
            result = -field
            neg_time = time.time() - start_time
            print(f"Negation time: {neg_time:.6f} seconds")

            # abs
            start_time = time.time()
            result = abs(field)
            abs_time = time.time() - start_time
            print(f"Absolute value time: {abs_time:.6f} seconds")

            # to_np
            start_time = time.time()
            np_data = field.to_np()
            to_np_time = time.time() - start_time
            print(f"Conversion to NumPy time: {to_np_time:.6f} seconds")


if __name__ == "__main__":
    with open("./tests/reports/report.txt", "w", encoding="utf8") as reporter:
        suit = unittest.TestLoader().loadTestsFromTestCase(TestFields)

        runner = unittest.TextTestRunner(stream=reporter, verbosity=2)
        runner.run(suit)
