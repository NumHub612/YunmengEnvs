from core.numerics.fields import (
    NodeField,
    CellField,
    FaceField,
    Scalar,
    Vector,
    VariableType,
)
from core.numerics.mesh import (
    Grid1D,
    Grid2D,
    Grid3D,
    Coordinate,
    Node,
    Mesh,
    MeshTopo,
    MeshGeom,
)
from core.viewer.plotter import MatPlotters
from core.solvers.commons import boundaries, inits, callbacks
from core.solvers import fvm
from core.utils.SympifyNumExpr import lambdify_numexpr

import matplotlib.pyplot as plt
import numpy as np
import os
import unittest
import shutil


class TestFvmEqs(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        print(f"\n---------- Testing < {cls.__name__} > \n")

    @classmethod
    def tearDownClass(cls):
        print("\n---------- Done \n")

    def setUp(self):
        self._output_dir = "./tests/results"

    def tearDown(self):
        pass

    def _plot_grid_index(self, grid: Grid2D, save_dir: str = "./tests/results"):
        """Plot the grid with node indices."""
        nx, ny = grid.nx, grid.ny

        for i in range(nx):
            for j in range(ny):
                nid = grid.match_node(i, j)
                coo = grid.nodes[nid].coordinate
                plt.text(coo.x, coo.y, str(nid), fontsize=8)
        plt.scatter(
            [n.coordinate.x for n in grid.nodes],
            [n.coordinate.y for n in grid.nodes],
            s=10,
        )
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("Node Coordinates")
        plt.savefig(os.path.join(save_dir, "node_coordinates.png"))
        # plt.show()
        plt.close()

        for f in grid.faces:
            fid = f.id
            coo = f.coordinate
            plt.text(coo.x, coo.y, str(fid), fontsize=8)
        plt.scatter(
            [f.coordinate.x for f in grid.faces],
            [f.coordinate.y for f in grid.faces],
            s=10,
        )
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("Face Coordinates")
        plt.savefig(os.path.join(save_dir, "face_coordinates.png"))
        # plt.show()
        plt.close()

        for c in grid.cells:
            cid = c.id
            coo = c.coordinate
            plt.text(coo.x, coo.y, str(cid), fontsize=8)
        plt.scatter(
            [c.coordinate.x for c in grid.cells],
            [c.coordinate.y for c in grid.cells],
            s=10,
        )
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("Cell Coordinates")
        plt.savefig(os.path.join(save_dir, "cell_coordinates.png"))
        # plt.show()
        plt.close()

        geom = MeshGeom(grid)
        normals = geom.face_normals
        x = [f.coordinate.x for f in grid.faces]
        y = [f.coordinate.y for f in grid.faces]
        vx = [n.x for n in normals]
        vy = [n.y for n in normals]
        plt.figure(figsize=(6, 5))
        plt.quiver(x, y, vx, vy, color="crimson", scale=None, width=0.005)
        plt.scatter(x, y, s=10)
        plt.gca().set_aspect("equal", adjustable="box")
        plt.savefig(os.path.join(save_dir, "face_normals.png"))
        # plt.show()
        plt.close()

    def _extract_boundaries(self, nx, ny, grid, topo):
        """Extract boundary faces and nodes."""
        bc_faces = topo.boundary_faces
        bc_groups = {}

        faces = []
        for j in range(ny - 1):
            cid = grid.match_cell(0, j)
            for fid in grid.cells[cid].faces:
                if fid in bc_faces:
                    faces.append(grid.faces[fid])
        nid = grid.match_node(0, 0)
        coo = grid.nodes[nid].coordinate
        for face in faces:
            if face.coordinate.x != coo.x:
                faces.remove(face)
        bc_groups["west"] = faces

        faces = []
        for j in range(ny - 1):
            cid = grid.match_cell(nx - 2, j)
            for fid in grid.cells[cid].faces:
                if fid in bc_faces:
                    faces.append(grid.faces[fid])
        nid = grid.match_node(nx - 1, 0)
        coo = grid.nodes[nid].coordinate
        for face in faces:
            if face.coordinate.x != coo.x:
                faces.remove(face)
        bc_groups["east"] = faces

        faces = []
        for i in range(nx - 1):
            cid = grid.match_cell(i, 0)
            for fid in grid.cells[cid].faces:
                if fid in bc_faces:
                    faces.append(grid.faces[fid])
        nid = grid.match_node(0, 0)
        coo = grid.nodes[nid].coordinate
        for face in faces:
            if face.coordinate.y != coo.y:
                faces.remove(face)
        bc_groups["south"] = faces

        faces = []
        for i in range(nx - 1):
            cid = grid.match_cell(i, ny - 2)
            for fid in grid.cells[cid].faces:
                if fid in bc_faces:
                    faces.append(grid.faces[fid])
        nid = grid.match_node(0, ny - 1)
        coo = grid.nodes[nid].coordinate
        for face in faces:
            if face.coordinate.y != coo.y:
                faces.remove(face)
        bc_groups["north"] = faces

        return bc_groups

    def test_diffusion_2d(self):
        """test Diffusion2D."""
        # set mesh
        low_left, upper_right = Coordinate(0, 0), Coordinate(0.833, 0.83)
        nx, ny = 11, 11
        grid = Grid2D(low_left, upper_right, nx, ny)
        topo = MeshTopo(grid)
        bc_groups = self._extract_boundaries(nx, ny, grid, topo)
        self._plot_grid_index(grid)

        # set initial condition
        ic = inits.UniformInitialization("ic1", Scalar(0.0))
        # cell_num = grid.cell_count
        # init_field = CellField(cell_num, VariableType.SCALAR)
        # ic = inits.HotstartInitialization("ic1", init_field)

        # set boundary condition
        bc1 = boundaries.FixedBoundary("bc1", 100)
        bc2 = boundaries.FixedBoundary("bc2", 20)

        # set callback
        output_dir = os.path.join(self._output_dir, "diff")
        confs = {
            "u": {
                "style": "cloudmap",
                "show_edges": True,
            }
        }
        cb1 = callbacks.ImageRender("cb1", output_dir, fields=confs)

        # set solver
        solver = fvm.Diffusion2D("solver1", grid)
        solver.add_callback(cb1)
        solver.add_ic("u", ic)
        solver.add_bc("u", bc_groups["west"], bc1)
        solver.add_bc("u", bc_groups["east"], bc1)
        solver.add_bc("u", bc_groups["south"], bc1)
        solver.add_bc("u", bc_groups["north"], bc2)

        K = 100
        solver.initialize(K)

        # run solver
        is_done = False
        while not is_done:
            is_done, _, _ = solver.inference()

        cb1.on_task_end()

    def test_diffusion_unsteady(self):
        """test unsteady Diffusion."""
        # set mesh
        low_left, upper_right = Coordinate(0, 0), Coordinate(0.833, 0.83)
        nx, ny = 31, 31
        grid = Grid2D(low_left, upper_right, nx, ny)
        topo = MeshTopo(grid)
        bc_groups = self._extract_boundaries(nx, ny, grid, topo)
        self._plot_grid_index(grid)

        # set initial condition
        ic = inits.UniformInitialization("ic1", Scalar(0.0))

        # set boundary condition
        bc1 = boundaries.FixedBoundary("bc1", 100)
        bc2 = boundaries.FixedBoundary("bc2", 20)

        # set callback
        output_dir = os.path.join(self._output_dir, "diff2")
        confs = {
            "phi": {
                "style": "cloudmap",
                "show_edges": True,
            }
        }
        cb1 = callbacks.ImageRender("cb1", output_dir, fields=confs)

        # set solver
        solver = fvm.UnsteadyDiffusion("solver1", grid)
        solver.add_callback(cb1)
        solver.add_ic("phi", ic)
        solver.add_bc("phi", bc_groups["west"], bc1)
        solver.add_bc("phi", bc_groups["east"], bc1)
        solver.add_bc("phi", bc_groups["south"], bc1)
        solver.add_bc("phi", bc_groups["north"], bc2)

        steps = 10
        K = 1
        dt = 0.01
        # run at first order accuracy
        solver.initialize(K, order=1, max_iter=steps)
        print("Running solver at first order accuracy...")

        status = None
        while status is None or not status.finished:
            status = solver.inference(dt)
        cb1.on_task_end()

        # run solver at second order accuracy
        solver.initialize(K, order=2, max_iter=steps)
        print("Running solver at second order accuracy...")

        status = None
        while status is None or not status.finished:
            status = solver.inference(dt)
        cb1.on_task_end()

    def test_convection_2d(self):
        """test Convection2D."""
        # set mesh
        low_left, upper_right = Coordinate(0, 0), Coordinate(1.0, 1.0)
        nx, ny = 11, 11
        grid = Grid2D(low_left, upper_right, nx, ny)
        topo = MeshTopo(grid)
        bc_groups = self._extract_boundaries(nx, ny, grid, topo)
        self._plot_grid_index(grid)

        # set initial condition
        ic1 = inits.UniformInitialization("ic1", Scalar(0.5))
        ic2 = inits.UniformInitialization("ic2", Vector(1.0, 1.0))

        # set boundary condition
        bc1 = boundaries.FixedBoundary("bc1", 0)
        bc2 = boundaries.FixedBoundary("bc2", 1)
        bc3 = boundaries.NaturalBoundary("bc3", Vector(0.0, 0.0))

        # set callback
        output_dir = os.path.join(self._output_dir, "conv")
        confs = {
            "phi": {
                "style": "cloudmap",
                "show_edges": True,
                "show_scalars": False,
            },
            "u": {"style": "streamplot"},
        }
        cb1 = callbacks.ImageRender("cb1", output_dir, fields=confs)

        # set solver
        solver = fvm.Convection2D("solver2", grid)
        solver.add_callback(cb1)
        solver.add_ic("phi", ic1)
        solver.add_ic("u", ic2)
        solver.add_bc("phi", bc_groups["west"], bc2)
        solver.add_bc("phi", bc_groups["east"], bc3)
        solver.add_bc("phi", bc_groups["south"], bc1)
        solver.add_bc("phi", bc_groups["north"], bc3)

        solver.initialize()

        # run solver
        done = False
        while not done:
            done, _, _ = solver.inference()

        cb1.on_task_end()

    def test_convection_unsteady(self):
        """test unsteady Convection."""
        # set mesh
        low_left, upper_right = Coordinate(0, 0), Coordinate(1.0, 1.0)
        nx, ny = 31, 31
        grid = Grid2D(low_left, upper_right, nx, ny)
        topo = MeshTopo(grid)
        bc_groups = self._extract_boundaries(nx, ny, grid, topo)
        self._plot_grid_index(grid)

        # set initial condition
        ic1 = inits.UniformInitialization("ic1", Scalar(0.5))

        # set boundary condition
        bc1 = boundaries.FixedBoundary("bc1", 0)
        bc2 = boundaries.FixedBoundary("bc2", 1)
        bc3 = boundaries.NaturalBoundary("bc3", Vector(0.0, 0.0))

        # set callback
        output_dir = os.path.join(self._output_dir, "conv2")
        confs = {
            "phi": {
                "style": "cloudmap",
                "show_edges": True,
                "show_scalars": False,
            },
        }
        cb1 = callbacks.ImageRender("cb1", output_dir, fields=confs)

        # set solver
        solver = fvm.UnsteadyConvection("solver2", grid)
        solver.add_callback(cb1)
        solver.add_ic("phi", ic1)
        solver.add_bc("phi", bc_groups["west"], bc2)
        solver.add_bc("phi", bc_groups["east"], bc3)
        solver.add_bc("phi", bc_groups["south"], bc1)
        solver.add_bc("phi", bc_groups["north"], bc3)

        steps = 10
        solver.initialize(max_iter=steps)

        # run solver
        dt = 0.1
        status = None
        while status is None or not status.finished:
            status = solver.inference(dt)

        cb1.on_task_end()

    def test_burgers(self):
        """test Burgers2D."""
        # set mesh
        low_left, upper_right = Coordinate(0, 0), Coordinate(2, 2)
        nx, ny = 21, 21
        grid = Grid2D(low_left, upper_right, nx, ny)

        start_x, end_x = 0.5, 1.0
        start_y, end_y = 0.5, 1.0
        init_groups = []
        for cell in grid.cells:
            if (
                cell.coordinate.x >= start_x
                and cell.coordinate.x <= end_x
                and cell.coordinate.y >= start_y
                and cell.coordinate.y <= end_y
            ):
                init_groups.append(cell.id)

        topo = MeshTopo(grid)
        bc_groups = self._extract_boundaries(nx, ny, grid, topo)

        # set initial condition
        cell_num = grid.cell_count
        init_field = CellField(cell_num, VariableType.VECTOR, Vector(1, 1))
        for i in init_groups:
            init_field[i] = Vector(2, 2)

        ic = inits.HotstartInitialization("ic1", init_field)

        # set boundary condition
        bc_value = Vector(1, 1)
        bc1 = boundaries.NaturalBoundary("bc1", bc_value)
        bc2 = boundaries.FixedBoundary("bc2", bc_value)

        # set callback
        output_dir = os.path.join(self._output_dir, "burgers")
        confs = {
            "u": {
                "style": "cloudmap",
                "dimension": "x",
                "show_edges": True,
                # "show_scalars": True,
            },
        }
        cb1 = callbacks.ImageRender("cb1", output_dir, confs)

        # set conditions
        solver = fvm.UnsteadyBurgers("solver1", grid)
        solver.add_callback(cb1)
        solver.add_ic("u", ic)
        solver.add_bc("u", bc_groups["west"], bc2)
        solver.add_bc("u", bc_groups["south"], bc2)
        solver.add_bc("u", bc_groups["east"], bc1)
        solver.add_bc("u", bc_groups["north"], bc1)

        # initialize solver
        steps = 10
        K = 0.1
        dt = 0.04
        order = 1
        solver.initialize(k=K, order=order, max_iter=steps)

        # run solver
        status = None
        while status is None or not status.finished:
            status = solver.inference(dt)

        cb1.on_task_end()


if __name__ == "__main__":
    with open("./tests/reports/report.txt", "w", encoding="utf8") as reporter:
        suit = unittest.TestSuite()
        # suit.addTest(TestFvmEqs("test_diffusion_2d"))
        # suit.addTest(TestFvmEqs("test_diffusion_unsteady"))
        # suit.addTest(TestFvmEqs("test_convection_2d"))
        # suit.addTest(TestFvmEqs("test_convection_unsteady"))
        suit.addTest(TestFvmEqs("test_burgers"))

        runner = unittest.TextTestRunner(stream=reporter, verbosity=2)
        runner.run(suit)
