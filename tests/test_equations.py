# -*- encoding: utf-8 -*-
from core.numerics.mesh import Grid2D, Coordinate, MeshTopo
from core.solvers.fdm.operators import fdm_operators
from core.solvers.commons import inits, boundaries, SimpleEquation
from core.numerics.fields import NodeField, Vector, Scalar
import numpy as np
import os
import shutil

import unittest


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
        pass

    def test_full_burgers2d(self):
        """test full burgers equation"""
        print("Testing full burgers equation...")
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
        eqs, solution = self._run(problem)

        diags = eqs.matrix.diag
        coef = 1 / self._dt
        self.assertTrue(np.all(diags == Vector.unit() * coef))

        validates = [
            Vector(24.999999999999996, 24.999999999999996, 0.0),
            Vector(24.999999999999996, 24.999999999999996, 0.0),
            Vector(24.999999999999996, 24.999999999999996, 0.0),
            Vector(24.999999999999996, 24.999999999999996, 0.0),
            Vector(24.999999999999996, 24.999999999999996, 0.0),
            Vector(24.999999999999996, 24.999999999999996, 0.0),
            Vector(24.999999999999996, 24.999999999999996, 0.0),
            Vector(24.999999999999996, 24.999999999999996, 0.0),
            Vector(24.999999999999996, 24.999999999999996, 0.0),
            Vector(24.999999999999996, 24.999999999999996, 0.0),
            Vector(24.999999999999996, 24.999999999999996, 0.0),
            Vector(24.999999999999996, 24.999999999999996, 0.0),
            Vector(25.047182787712256, 25.047182787712256, 0.0),
            Vector(25.137404443691608, 25.137404443691608, 0.0),
            Vector(25.272073881050446, 25.272073881050446, 0.0),
            Vector(25.428452349685998, 25.428452349685998, 0.0),
            Vector(25.576297609182422, 25.576297609182422, 0.0),
            Vector(25.660527038990995, 25.660527038990995, 0.0),
            Vector(25.649976326387684, 25.649976326387684, 0.0),
            Vector(25.529566297465944, 25.529566297465944, 0.0),
            Vector(25.32693771181701, 25.32693771181701, 0.0),
            Vector(24.999999999999996, 24.999999999999996, 0.0),
            Vector(24.999999999999996, 24.999999999999996, 0.0),
            Vector(25.137404443691608, 25.137404443691608, 0.0),
            Vector(25.401089618988312, 25.401089618988312, 0.0),
            Vector(25.78121859620961, 25.78121859620961, 0.0),
            Vector(26.236060382789333, 26.236060382789333, 0.0),
            Vector(26.64339115794685, 26.64339115794685, 0.0),
            Vector(26.907046346116353, 26.907046346116353, 0.0),
            Vector(26.886528537664155, 26.886528537664155, 0.0),
            Vector(26.558121231710448, 26.558121231710448, 0.0),
            Vector(25.977622125457184, 25.977622125457184, 0.0),
            Vector(24.999999999999996, 24.999999999999996, 0.0),
            Vector(24.999999999999996, 24.999999999999996, 0.0),
            Vector(25.272073881050446, 25.272073881050446, 0.0),
            Vector(25.78121859620961, 25.78121859620961, 0.0),
            Vector(26.52611731639583, 26.52611731639583, 0.0),
            Vector(27.372313085999394, 27.372313085999394, 0.0),
            Vector(28.189770136046455, 28.189770136046455, 0.0),
            Vector(28.688604043685036, 28.688604043685036, 0.0),
            Vector(28.705650590056365, 28.705650590056365, 0.0),
            Vector(28.114093397542554, 28.114093397542554, 0.0),
            Vector(26.96099260391062, 26.96099260391062, 0.0),
            Vector(24.999999999999996, 24.999999999999996, 0.0),
            Vector(24.999999999999996, 24.999999999999996, 0.0),
            Vector(25.428452349685998, 25.428452349685998, 0.0),
            Vector(26.236060382789333, 26.236060382789333, 0.0),
            Vector(27.372313085999394, 27.372313085999394, 0.0),
            Vector(28.726628831499532, 28.726628831499532, 0.0),
            Vector(29.95144062390703, 29.95144062390703, 0.0),
            Vector(30.803539548142073, 30.803539548142073, 0.0),
            Vector(30.89279828423506, 30.89279828423506, 0.0),
            Vector(29.94544886750881, 29.94544886750881, 0.0),
            Vector(28.16839623124776, 28.16839623124776, 0.0),
            Vector(24.999999999999996, 24.999999999999996, 0.0),
            Vector(24.999999999999996, 24.999999999999996, 0.0),
            Vector(25.576297609182422, 25.576297609182422, 0.0),
            Vector(26.64339115794685, 26.64339115794685, 0.0),
            Vector(28.189770136046455, 28.189770136046455, 0.0),
            Vector(29.95144062390703, 29.95144062390703, 0.0),
            Vector(31.66267024884109, 31.66267024884109, 0.0),
            Vector(32.84202349326511, 32.84202349326511, 0.0),
            Vector(32.87766278768736, 32.87766278768736, 0.0),
            Vector(31.7383360200947, 31.7383360200947, 0.0),
            Vector(29.119852960027718, 29.119852960027718, 0.0),
            Vector(24.999999999999996, 24.999999999999996, 0.0),
            Vector(24.999999999999996, 24.999999999999996, 0.0),
            Vector(25.660527038990995, 25.660527038990995, 0.0),
            Vector(26.907046346116353, 26.907046346116353, 0.0),
            Vector(28.688604043685036, 28.688604043685036, 0.0),
            Vector(30.803539548142073, 30.803539548142073, 0.0),
            Vector(32.84202349326511, 32.84202349326511, 0.0),
            Vector(34.08859212391543, 34.08859212391543, 0.0),
            Vector(34.34295871393793, 34.34295871393793, 0.0),
            Vector(32.44085746900855, 32.44085746900855, 0.0),
            Vector(29.490145250009913, 29.490145250009913, 0.0),
            Vector(24.999999999999996, 24.999999999999996, 0.0),
            Vector(24.999999999999996, 24.999999999999996, 0.0),
            Vector(25.649976326387684, 25.649976326387684, 0.0),
            Vector(26.886528537664155, 26.886528537664155, 0.0),
            Vector(28.705650590056365, 28.705650590056365, 0.0),
            Vector(30.89279828423506, 30.89279828423506, 0.0),
            Vector(32.87766278768736, 32.87766278768736, 0.0),
            Vector(34.34295871393793, 34.34295871393793, 0.0),
            Vector(33.848845707725886, 33.848845707725886, 0.0),
            Vector(31.934278095873776, 31.934278095873776, 0.0),
            Vector(28.616798035995, 28.616798035995, 0.0),
            Vector(24.999999999999996, 24.999999999999996, 0.0),
            Vector(24.999999999999996, 24.999999999999996, 0.0),
            Vector(25.529566297465944, 25.529566297465944, 0.0),
            Vector(26.558121231710448, 26.558121231710448, 0.0),
            Vector(28.114093397542554, 28.114093397542554, 0.0),
            Vector(29.94544886750881, 29.94544886750881, 0.0),
            Vector(31.738336020094696, 31.738336020094696, 0.0),
            Vector(32.44085746900855, 32.44085746900855, 0.0),
            Vector(31.934278095873776, 31.934278095873776, 0.0),
            Vector(29.588859279166204, 29.588859279166204, 0.0),
            Vector(27.12146096444272, 27.12146096444272, 0.0),
            Vector(24.999999999999996, 24.999999999999996, 0.0),
            Vector(24.999999999999996, 24.999999999999996, 0.0),
            Vector(25.32693771181701, 25.32693771181701, 0.0),
            Vector(25.977622125457184, 25.977622125457184, 0.0),
            Vector(26.96099260391062, 26.96099260391062, 0.0),
            Vector(28.16839623124776, 28.16839623124776, 0.0),
            Vector(29.119852960027718, 29.119852960027718, 0.0),
            Vector(29.490145250009913, 29.490145250009913, 0.0),
            Vector(28.616798035995, 28.616798035995, 0.0),
            Vector(27.12146096444272, 27.12146096444272, 0.0),
            Vector(25.813825800293884, 25.813825800293884, 0.0),
            Vector(24.999999999999996, 24.999999999999996, 0.0),
            Vector(24.999999999999996, 24.999999999999996, 0.0),
            Vector(24.999999999999996, 24.999999999999996, 0.0),
            Vector(24.999999999999996, 24.999999999999996, 0.0),
            Vector(24.999999999999996, 24.999999999999996, 0.0),
            Vector(24.999999999999996, 24.999999999999996, 0.0),
            Vector(24.999999999999996, 24.999999999999996, 0.0),
            Vector(24.999999999999996, 24.999999999999996, 0.0),
            Vector(24.999999999999996, 24.999999999999996, 0.0),
            Vector(24.999999999999996, 24.999999999999996, 0.0),
            Vector(24.999999999999996, 24.999999999999996, 0.0),
            Vector(24.999999999999996, 24.999999999999996, 0.0),
        ]
        self.assertEqual(len(solution), len(validates))
        for i in range(len(solution)):
            self.assertTrue(np.allclose(solution[i] * coef, validates[i].data))

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

    def test_grad(self):
        """test grad equation"""
        print("Testing grad equation...")
        # set equations
        equation_expr = "u*grad::Grad01(u) == 0"
        symbols = {
            "u": {
                "description": "velocity",
                "coefficient": False,
                "type": "vector",
                "bounds": (None, None),
            }
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

    def test_laplacian(self):
        """test laplacian equation"""
        print("Testing laplacian equation...")
        # set equations
        equation_expr = "0 == nu*laplacian::Lap01(u)"
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

    def _run(self, problem):
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
