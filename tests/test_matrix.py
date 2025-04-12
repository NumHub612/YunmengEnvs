from core.numerics.matrix import LinearEqs
from core.numerics.matrix import DenseMatrix, SparseMatrix, SciMatrix
from core.numerics.fields import Field
import numpy as np
import time
import os
import subprocess
import unittest


class TestDenseMatrixes(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        print(f"\n---------- Testing < {cls.__name__} > \n")

        output_dir = "./tests/results"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        profile = os.path.join(output_dir, f"test_matrix_perf.svg")
        # subprocess.Popen(["py-spy", "record", "-o", profile, "--pid", str(os.getpid())])

    @classmethod
    def tearDownClass(cls):
        print("\n---------- Done \n")

    def setUp(self):
        self._output_dir = "./tests/results"
        self._scales = [10, 100, 1000]

    def tearDown(self):
        pass

    def test_baseline_matrix(self):
        print("test basline matrix operations: \n")

        for size in self._scales:
            self._run_matrix_ops(size, "float")

    def test_scalar_matrix(self):
        print("test scalar matrix operations: \n")

        for size in self._scales:
            self._run_matrix_ops(size, "scalar")

    def test_vector_matrix(self):
        print("test vector matrix operations: \n")

        for size in self._scales:
            self._run_matrix_ops(size, "vector")

    def test_tensor_matrix(self):
        print("test tensor matrix operations: \n")

        for size in self._scales:
            self._run_matrix_ops(size, "tensor")

    def _run_matrix_ops(self, size, type):
        print(f"--- {type} equations ({size}*{size}) operation perf:")
        mat1 = DenseMatrix.identity((size, size), type)
        mat2 = DenseMatrix.identity((size, size), type)

        start = time.time()
        res = mat1 + mat2
        end = time.time()
        print(f"+ add op(s): {end-start}")

        start = end
        res = mat1 - mat2
        end = time.time()
        print(f"+ sub op(s): {end-start}")

        start = end
        res = 1.5 * mat1
        end = time.time()
        print(f"+ mul op(s): {end-start}")

        start = end
        res = mat1 / 2.0
        end = time.time()
        print(f"+ div op(s): {end-start}")


class TestSparseMatrixes(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        print(f"\n---------- Testing < {cls.__name__} > \n")

        output_dir = "./tests/results"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        profile = os.path.join(output_dir, f"test_matrix_perf.svg")
        # subprocess.Popen(["py-spy", "record", "-o", profile, "--pid", str(os.getpid())])

    @classmethod
    def tearDownClass(cls):
        print("\n---------- Done \n")

    def setUp(self):
        self._output_dir = "./tests/results"
        self._scales = [100, 1000, 10000, 100000, 1000000]

    def tearDown(self):
        pass

    def test_baseline_matrix(self):
        print("test basline matrix operations: \n")

        for size in self._scales:
            self._run_matrix_ops(size, "float")

    def test_scalar_matrix(self):
        print("test scalar matrix operations: \n")

        for size in self._scales:
            self._run_matrix_ops(size, "scalar")

    def test_vector_matrix(self):
        print("test vector matrix operations: \n")

        for size in self._scales:
            self._run_matrix_ops(size, "vector")

    def test_tensor_matrix(self):
        print("test tensor matrix operations: \n")

        for size in self._scales:
            self._run_matrix_ops(size, "tensor")

    def _run_matrix_ops(self, size, type):
        print(f"--- {type} equations ({size}*{size}) operation perf:")
        mat1 = SparseMatrix.identity((size, size), type)
        mat2 = SparseMatrix.identity((size, size), type)

        start = time.time()
        res = mat1 + mat2
        end = time.time()
        print(f"+ add op(s): {end-start}")

        start = end
        res = mat1 - mat2
        end = time.time()
        print(f"+ sub op(s): {end-start}")

        start = end
        res = 1.5 * mat1
        end = time.time()
        print(f"+ mul op(s): {end-start}")

        start = end
        res = mat1 / 2.0
        end = time.time()
        print(f"+ div op(s): {end-start}")


class TestSciMatrixes(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        print(f"\n---------- Testing < {cls.__name__} > \n")

        output_dir = "./tests/results"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        profile = os.path.join(output_dir, f"test_matrix_perf.svg")
        # subprocess.Popen(["py-spy", "record", "-o", profile, "--pid", str(os.getpid())])

    @classmethod
    def tearDownClass(cls):
        print("\n---------- Done \n")

    def setUp(self):
        self._output_dir = "./tests/results"
        self._scales = [100, 1000, 10000, 100000, 1000000]

    def tearDown(self):
        pass

    def test_baseline_matrix(self):
        print("test basline matrix operations: \n")

        for size in self._scales:
            self._run_matrix_ops(size, "float")

    def _run_matrix_ops(self, size, type):
        print(f"--- {type} equations ({size}*{size}) operation perf:")
        mat1 = SciMatrix.identity((size, size), type)
        mat2 = SciMatrix.identity((size, size), type)

        start = time.time()
        res = mat1 + mat2
        end = time.time()
        print(f"+ add op(s): {end-start}")

        start = end
        res = mat1 - mat2
        end = time.time()
        print(f"+ sub op(s): {end-start}")

        start = end
        res = 1.5 * mat1
        end = time.time()
        print(f"+ mul op(s): {end-start}")

        start = end
        res = mat1 / 2.0
        end = time.time()
        print(f"+ div op(s): {end-start}")


if __name__ == "__main__":
    with open("./tests/reports/report.txt", "w", encoding="utf8") as reporter:
        suit = unittest.TestSuite()
        # suit.addTest(TestDenseMatrixes("test_baseline_matrix"))
        # suit.addTest(TestDenseMatrixes("test_scalar_matrix"))
        # suit.addTest(TestDenseMatrixes("test_vector_matrix"))
        # suit.addTest(TestDenseMatrixes("test_tensor_matrix"))
        # suit.addTest(TestSparseMatrixes("test_baseline_matrix"))
        # suit.addTest(TestSparseMatrixes("test_scalar_matrix"))
        # suit.addTest(TestSparseMatrixes("test_vector_matrix"))
        # suit.addTest(TestSparseMatrixes("test_tensor_matrix"))
        suit.addTest(TestSciMatrixes("test_baseline_matrix"))

        runner = unittest.TextTestRunner(stream=reporter, verbosity=2)
        runner.run(suit)
