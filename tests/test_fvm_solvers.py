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
from core.renders.plotter import MatPlotters
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
        normals = geom.face_normal
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
        bc_value = np.array([1, 1, 0])
        bc1 = boundaries.NaturalBoundary("bc1", bc_value)
        bc2 = boundaries.FixedBoundary("bc2", bc_value)

        # set callback
        output_dir = os.path.join(self._output_dir, "burgers")
        cb2 = callbacks.PerformanceMonitor("burgers2d", output_dir, 1)

        confs = {
            "u": {
                "style": "cloudmap",
                "dimension": "x",
                "show_edges": True,
                # "show_scalars": True,
            },
        }
        cb1 = callbacks.ImageRender("cb1", output_dir, confs)

        # set operators
        operators = {
            "ddt": fvm.fvm_operators["ddt01"](rho=1.0),
            "grad": fvm.fvm_operators["grad01"](),
            "div": fvm.fvm_operators["div01"](rho=1.0),
            "laplacian": fvm.fvm_operators["lap01"](k=0.1),
            "src": fvm.fvm_operators["src01"](),
        }

        # set conditions
        solver = fvm.Burgers2D("solver1", grid, operators)
        solver.add_callback(cb1)
        solver.add_callback(cb2)
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
        solver.initialize(max_iter=steps)

        # run solver
        status = None
        while status is None or not status.finished:
            status = solver.inference(dt)

        cb1.on_task_end()


if __name__ == "__main__":
    with open("./tests/reports/report.txt", "w", encoding="utf8") as reporter:
        suit = unittest.TestSuite()
        suit.addTest(TestFvmEqs("test_burgers"))

        runner = unittest.TextTestRunner(stream=reporter, verbosity=2)
        runner.run(suit)
