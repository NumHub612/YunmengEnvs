from core.numerics.mesh import Grid1D, Grid2D, Grid3D, Coordinate, Node, Mesh
from core.numerics.mesh import MeshTopo
from core.numerics.fields import NodeField, Scalar, Vector
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
        cb2.on_task_begin()

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

        cb2.on_task_end()

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
        # set mesh
        low_left, upper_right = Coordinate(0, 0), Coordinate(2, 2)
        nx, ny = 41, 41
        grid = Grid2D(low_left, upper_right, nx, ny)

        dx = 2 / (nx - 1)
        dy = 2 / (ny - 1)

        start_x, end_x = int(0.5 / dx), int(1.0 / dx) + 1
        start_y, end_y = int(0.5 / dy), int(1.0 / dy) + 1
        init_groups = []
        for i in range(start_x, end_x):
            for j in range(start_y, end_y):
                index = i * ny + j
                init_groups.append(index)

        topo = MeshTopo(grid)
        bc_groups = []
        for i in topo.boundary_nodes_indices:
            bc_groups.append(grid.nodes[i])

        # set initial condition
        node_num = grid.node_count
        init_field = NodeField(node_num, "vector", Vector(1, 1))
        for i in init_groups:
            init_field[i] = Vector(2, 2)

        ic = inits.HotstartInitialization("ic1", init_field)

        # set boundary condition
        bc_value = Vector(1, 1)
        bc = boundaries.ConstantBoundary("bc1", bc_value, None)

        # set callback
        output_dir = "./tests/results"
        confs = {
            "vel": {"style": "cloudmap", "dimension": "x"},
        }
        cb1 = callbacks.RenderCallback(output_dir, confs)
        cb2 = callbacks.PerformanceMonitor(
            "burgers2d", "./tests/results/burgers2d.log", 10
        )

        # set solver
        solver = fdm.Burgers2D("solver1", grid)
        solver.add_callback(cb1)
        solver.add_callback(cb2)
        solver.add_ic("vel", ic)
        solver.add_bc("vel", bc_groups, bc)

        sigma = 0.2
        dt = sigma * dx
        nb_steps = 60
        solver.initialize(nb_steps, sigma=sigma)
        cb2.on_task_begin()

        # run solver
        is_done = False
        while not is_done:
            is_done, _, _ = solver.inference(dt)

        cb2.on_task_end()

    def test_burgers_3d(self):
        # set mesh
        low_left, upper_right = Coordinate(0, 0, 0), Coordinate(2, 2, 2)
        nx, ny, nz = 11, 11, 11
        grid = Grid3D(low_left, upper_right, nx, ny, nz)
        topo = MeshTopo(grid)

        node_index_groups = []
        for i in range(9, 11):
            for j in range(2):
                for k in range(2):
                    node_index_groups.append(grid.match_node(i, j, k))

        bc_node_groups = []
        for b in topo.boundary_nodes_indices:
            bc_node_groups.append(grid.nodes[b])

        # set initial condition
        node_num = grid.node_count
        init_field = NodeField(node_num, "vector", Vector(0, 0, 0))
        for i in node_index_groups:
            init_field[i] = Vector(-2, 2, 2)

        ic = inits.HotstartInitialization("ic1", init_field)

        # set boundary condition
        bc_value = Vector(0, 0, 0)
        bc = boundaries.ConstantBoundary("bc1", bc_value, None)

        # set callback
        output_dir = "./tests/results"
        confs = {
            "vel": {
                "style": "cloudmap",
                "dimension": "x",
                # "slice_set": {"style": "slice_along_axis", "n": 6, "axis": "y"},
                "slice_set": {"style": "slice", "normal": [1, 1, 0]},
            }
        }
        cb1 = callbacks.RenderCallback(output_dir, confs)
        cb2 = callbacks.PerformanceMonitor(
            "burgers3d", "./tests/results/burgers3d.log", 10
        )

        # set solver
        solver = fdm.Burgers3D("solver1", grid)
        solver.add_callback(cb1)
        solver.add_callback(cb2)
        solver.add_ic("vel", ic)
        solver.add_bc("vel", bc_node_groups, bc)

        sigma = 0.2
        nu = 0.02
        dx = 2 / grid.nx
        dt = sigma * dx
        nb_steps = 60
        solver.initialize(nb_steps, nu)
        cb2.on_task_begin()

        # run solver
        is_done = False
        while not is_done:
            is_done, _, status = solver.inference(dt)
            print(f"Step {status['current_time']}")

        cb2.on_task_end()


if __name__ == "__main__":
    with open("./tests/reports/report.txt", "w", encoding="utf8") as reporter:
        suit = unittest.TestSuite()
        suit.addTest(TestBurgers("test_burgers_1d"))
        # suit.addTest(TestBurgers("test_burgers_2d"))
        # suit.addTest(TestBurgers("test_burgers_3d"))

        runner = unittest.TextTestRunner(stream=reporter, verbosity=2)
        runner.run(suit)
