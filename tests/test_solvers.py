from core.numerics.mesh import Grid1D, Coordinate, Node, Mesh
from core.numerics.fields import NodeField, Scalar
from core.solvers.commons import boundaries, inits, callbacks
from core.solvers import fdm
from core.utils.SympifyNumExpr import lambdify_numexpr

import matplotlib.pyplot as plt
import numpy as np
import unittest


class TestBurgers(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        print(f"\n---------- Testing < {cls.__name__} > \n")

    @classmethod
    def tearDownClass(cls):
        print("\n---------- Done \n")

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_burgers_1d(self):
        # set mesh
        start, end = Coordinate(0), Coordinate(2 * np.pi)
        grid = Grid1D(start, end, 401)

        xs = np.linspace(0, 2 * np.pi, 401)
        group1 = [grid.nodes[0], grid.nodes[400]]

        # set initial condition
        phi = "exp(-(-4*t + x - 2*pi)**2/(4*nu*(t + 1))) + exp(-(-4*t + x)**2/(4*nu*(t + 1)))"
        phiprime = "-(-8*t + 2*x)*exp(-(-4*t + x)**2/(4*nu*(t + 1)))/(4*nu*(t + 1)) - (-8*t + 2*x - 4*pi)*exp(-(-4*t + x - 2*pi)**2/(4*nu*(t + 1)))/(4*nu*(t + 1))"
        expr = f"-2 * nu * (({phiprime}) / ({phi})) + 4"
        symbols = ["t", "x", "nu"]

        func = lambdify_numexpr(expr, symbols)
        nu = 0.07

        def ic_func(mesh: Mesh):
            field = NodeField(mesh.node_count, "scalar")
            for i in range(mesh.node_count):
                x = mesh.nodes[i].coordinate.x
                value = Scalar(func(0.0, x, nu))
                field[i] = value
            return field

        ic = inits.CustomInitialization("ic1", grid, ic_func)

        # set boundary condition
        def bc_func(t, node: Node):
            value = Scalar(func(t, node.coordinate.x, nu))
            return None, value

        bc = boundaries.CustomBoundary("bc1", bc_func)

        # set callback
        output_dir = "./tests/results"
        cb1 = callbacks.RenderCallback(output_dir)
        cb2 = callbacks.PerformanceMonitor(
            "burgers1d", "./tests/results/burgers1d.log", 10
        )

        # set solver
        solver = fdm.Burgers1D("solver1", grid)
        solver.add_callback(cb1)
        solver.add_callback(cb2)
        solver.add_ic("u", ic)
        solver.add_bc("u", group1, bc)

        nb_steps = 100
        solver.initialize(nb_steps)

        # analytical solution
        nx = 401
        dx = 2 * np.pi / (nx - 1)
        nu = 0.07
        dt = dx * nu

        total_time = nb_steps * dt

        # run solver
        is_done = False
        while not is_done:
            is_done, _, _ = solver.inference(dt)

        # get solution
        u_simu = solver.get_solution("u")
        u_simu = np.asarray([var.value for var in u_simu])
        u_real = np.asarray([func(total_time, x, nu) for x in xs])

        # plot solution
        plt.plot(xs, u_real, label="Analytical solution")
        plt.plot(xs, u_simu, label="Numerical solution")
        plt.legend()
        plt.xlabel("x")
        plt.ylabel("u")
        plt.title("Burgers equation")
        plt.savefig("./tests/results/burgers1d.png")
        plt.close()

    def test_burgers_2d(self):
        pass

    def test_burgers_3d(self):
        pass


if __name__ == "__main__":
    with open("./tests/reports/report.txt", "w", encoding="utf8") as reporter:
        suit = unittest.TestSuite()
        suit.addTest(TestBurgers("test_burgers_1d"))
        suit.addTest(TestBurgers("test_burgers_2d"))
        suit.addTest(TestBurgers("test_burgers_3d"))

        runner = unittest.TextTestRunner(stream=reporter, verbosity=2)
        runner.run(suit)
