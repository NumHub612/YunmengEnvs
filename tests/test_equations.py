# -*- encoding: utf-8 -*-
from core.numerics.mesh import Grid2D, Coordinate, MeshTopo
from core.solvers.fdm.operators import fdm_operators
from core.solvers.commons import inits, boundaries, SimpleEquation
from core.numerics.fields import NodeField, Vector, Scalar
from core.visuals.plotter import plot_field
from core.visuals.animator import ImageSetPlayer
import numpy as np
import os
import shutil
import unittest
import subprocess


class TestSimpleEquations(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        print(f"\n---------- Testing < {cls.__name__} > \n")

    @classmethod
    def tearDownClass(cls):
        print("\n---------- Done \n")

    def setUp(self):
        # set mesh
        low_left, upper_right = Coordinate(0, 0), Coordinate(2, 2)
        nx, ny = 11, 11
        self._grid = Grid2D(low_left, upper_right, nx, ny)
        self._topo = MeshTopo(self._grid)

        dx = 2 / (nx - 1)
        dy = 2 / (ny - 1)
        start_x, end_x = int(0.5 / dx), int(1.0 / dx) + 1
        start_y, end_y = int(0.5 / dy), int(1.0 / dy) + 1

        # set time step
        sigma = 0.2
        self._dt = sigma * dx

        # set initial condition
        init_groups = []
        for i in range(start_x, end_x):
            for j in range(start_y, end_y):
                index = i * ny + j
                init_groups.append(index)

        node_num = self._grid.node_count
        self._var_field = NodeField(node_num, "vector", Vector(1, 1), "u")
        for i in init_groups:
            self._var_field[i] = Vector(2, 2)

        self._ic = inits.HotstartInitialization("ic1", self._var_field)

        # set boundary condition
        bc_groups = []
        for i in self._topo.boundary_nodes_indexes:
            bc_groups.append(self._grid.nodes[i])

        bc_value = Vector(1, 1)
        self._bc = boundaries.ConstantBoundary("bc1", bc_value, None)

    def tearDown(self):
        # if os.path.exists("./tests/results/"):
        #     shutil.rmtree("./tests/results/")
        # os.makedirs("./tests/results/")
        pass

    def test_full_burgers2d(self):
        """test full burgers equation"""
        print("Testing full burgers equation...")

        # run performance profiling
        output_dir = "./tests/results"
        profile = os.path.join(output_dir, f"test_simple_equations_perf.svg")
        # subprocess.Popen(["py-spy", "record", "-o", profile, "--pid", str(os.getpid())])

        # set result output path
        save_dir = "./tests/results/u"
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)
        os.makedirs(save_dir)

        # set equations
        equation_expr = "ddt::Ddt01(u) + u*grad::Grad01(u) == nu*laplacian::Lap01(u)"
        symbols = {
            "u": {
                "description": "velocity",
                "coefficient": False,
                "type": "vector",
                "bounds": (None, None),
            },
            "nu": {
                "description": "viscosity",
                "coefficient": True,
                "type": "scalar",
                "bounds": (0, None),
            },
        }
        coefficients = {"nu": Scalar(0.1)}
        variables = {"u": self._var_field}

        # set problem
        problem = SimpleEquation("burgers2d", fdm_operators)
        problem.set_equations([equation_expr], symbols)
        problem.set_coefficients(coefficients)
        problem.set_fields(variables)
        problem.set_mesh(self._grid)

        # discretize and solve
        eqs, _ = self._run(problem, save_dir, True)

        diags = eqs.matrix.diag
        coef = 1 / self._dt
        self.assertTrue(np.all(diags == coef))

        # play the images
        ani = ImageSetPlayer(save_dir)
        if __name__ == "__main__":
            ani.play()

    def test_grad_burgers2d(self):
        """test burgers equation without laplacian term"""
        print("Testing burgers equation without laplacian term...")
        # set equations
        equation_expr = "ddt::Ddt01(u) + u*grad::Grad01(u) == 0"
        symbols = {
            "u": {
                "description": "velocity",
                "coefficient": False,
                "type": "vector",
                "bounds": (None, None),
            },
            "nu": {
                "description": "viscosity",
                "coefficient": True,
                "type": "scalar",
                "bounds": (0, None),
            },
        }
        coefficients = {"nu": Scalar(0.1)}
        variables = {"u": self._var_field}

        # set problem
        problem = SimpleEquation("burgers2d", fdm_operators)
        problem.set_equations([equation_expr], symbols)
        problem.set_coefficients(coefficients)
        problem.set_fields(variables)
        problem.set_mesh(self._grid)

        # discretize and solve
        self._run(problem)

    def test_lap_burgers2d(self):
        """test burgers equation without grad term"""
        # set equations
        equation_expr = "ddt::Ddt01(u) == nu*laplacian::Lap01(u)"
        symbols = {
            "u": {
                "description": "velocity",
                "coefficient": False,
                "type": "vector",
                "bounds": (None, None),
            },
            "nu": {
                "description": "viscosity",
                "coefficient": True,
                "type": "scalar",
                "bounds": (0, None),
            },
        }
        coefficients = {"nu": Scalar(0.1)}
        variables = {"u": self._var_field}

        # set problem
        problem = SimpleEquation("burgers2d", fdm_operators)
        problem.set_equations([equation_expr], symbols)
        problem.set_coefficients(coefficients)
        problem.set_fields(variables)
        problem.set_mesh(self._grid)

        # discretize and solve
        self._run(problem)

    def test_ddt(self):
        """test ddt equation"""
        print("Testing ddt equation...")
        # set equations
        equation_expr = "ddt::Ddt01(u) == 0"
        symbols = {
            "u": {
                "description": "velocity",
                "coefficient": False,
                "type": "vector",
                "bounds": (None, None),
            },
        }
        coefficients = {}
        variables = {"u": self._var_field}

        # set problem
        problem = SimpleEquation("burgers2d", fdm_operators)
        problem.set_equations([equation_expr], symbols)
        problem.set_coefficients(coefficients)
        problem.set_fields(variables)
        problem.set_mesh(self._grid)

        # discretize and solve
        self._run(problem)

    def _run(self, problem, result_dir=None, show=False):
        # set solving parameters
        steps = 10
        i = 0

        # solve
        eqs, solution = None, None
        while i < steps:
            # solve linear equation
            eqs = problem.discretize(self._dt)
            solution = eqs.solve()
            self._var_field = NodeField.from_np(solution, "node", "u")

            # show solution
            if __name__ == "__main__":
                if show and result_dir:
                    plot_field(
                        self._var_field,
                        self._grid,
                        title=f"u-{i}",
                        style="cloudmap",
                        dimension="x",
                        save_dir=result_dir,
                    )

            # update boundary condition
            for node in self._topo.boundary_nodes_indexes:
                _, val = self._bc.evaluate(None, None)
                self._var_field[node] = val

            # update initial condition
            problem.set_fields({"u": self._var_field})
            i += 1

        return eqs, solution


if __name__ == "__main__":
    with open("./tests/reports/report.txt", "w", encoding="utf8") as reporter:
        suit = unittest.TestSuite()
        suit.addTest(TestSimpleEquations("test_full_burgers2d"))

        # suit = unittest.TestLoader().loadTestsFromTestCase(TestSimpleEquations)

        runner = unittest.TextTestRunner(stream=reporter, verbosity=2)
        runner.run(suit)
