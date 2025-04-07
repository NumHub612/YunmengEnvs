from core.numerics.matrix import Matrix, LinearEqs
from core.numerics.fields import Scalar, Vector, Tensor, Field
import numpy as np
import time
import os
import subprocess
import unittest


class TestMatrixes(unittest.TestCase):

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

    def tearDown(self):
        pass

    def test_baseline_matrix(self):
        print("test basline matrix operations: \n")

        for size in [100, 1000, 10000, 100000, 1000000]:
            self._run_lineareqs_ops(size, "float")

    def test_scalar_matrix(self):
        print("test scalar matrix operations: \n")

        for size in [100, 1000, 10000, 100000, 1000000]:
            self._run_lineareqs_ops(size, "scalar")

    def test_vector_matrix(self):
        print("test vector matrix operations: \n")

        for size in [100, 1000, 10000, 100000, 1000000]:
            self._run_lineareqs_ops(size, "vector")

    def test_tensor_matrix(self):
        print("test tensor matrix operations: \n")

        for size in [100, 1000, 10000, 100000, 1000000]:
            self._run_lineareqs_ops(size, "tensor")

    def _run_lineareqs_ops(self, size, type):
        print(f"--- {type} equations ({size}*{size}) operation perf:")
        eq1 = LinearEqs.zeros("var1", size, type)
        eq2 = LinearEqs.zeros("var1", size, type)
        eq2._mat = Matrix.unit((size, size), type)
        eq2._rhs = Matrix.ones((size,), type)

        start = time.time()
        res = eq1 + eq2
        end = time.time()
        print(f"+ add op(s): {end-start}")

        start = end
        res = eq1 - eq2
        end = time.time()
        print(f"+ sub op(s): {end-start}")

        start = end
        res = 1.5 * eq1
        end = time.time()
        print(f"+ mul op(s): {end-start}")

        start = end
        res = eq1 / 2.0
        end = time.time()
        print(f"+ div op(s): {end-start}")


if __name__ == "__main__":
    with open("./tests/reports/report.txt", "w", encoding="utf8") as reporter:
        suit = unittest.TestSuite()
        suit.addTest(TestMatrixes("test_baseline_matrix"))
        suit.addTest(TestMatrixes("test_scalar_matrix"))
        suit.addTest(TestMatrixes("test_vector_matrix"))
        # suit.addTest(TestMatrixes("test_tensor_matrix"))

        runner = unittest.TextTestRunner(stream=reporter, verbosity=2)
        runner.run(suit)
