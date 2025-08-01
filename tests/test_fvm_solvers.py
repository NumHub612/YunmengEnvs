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
from core.numerics.fields import (
    NodeField,
    CellField,
    FaceField,
    Scalar,
    Vector,
    VariableType,
)
from core.solvers.commons import boundaries, inits, callbacks
from core.solvers import fvm
from core.utils.SympifyNumExpr import lambdify_numexpr

import matplotlib.pyplot as plt
import numpy as np
import os
import unittest


class TestFvmEqs(unittest.TestCase):

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

    def test_diffusion_2d(self):
        """test Diffusion2D."""
        # set mesh
        low_left, upper_right = Coordinate(0, 0), Coordinate(0.833, 0.83)
        nx, ny = 11, 11
        grid = Grid2D(low_left, upper_right, nx, ny)
        topo = MeshTopo(grid)

        self._plot_grid_index(grid)

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

        # set initial condition
        ic = inits.UniformInitialization("ic1", Scalar(0.0))
        # cell_num = grid.cell_count
        # init_field = CellField(cell_num, VariableType.SCALAR)
        # ic = inits.HotstartInitialization("ic1", init_field)

        # set boundary condition
        bc1 = boundaries.DirichletBoundary("bc1", 100)
        bc2 = boundaries.DirichletBoundary("bc2", 20)

        # set callback
        output_dir = "./tests/results"
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

    def test_convection_2d(self):
        """test Convection2D."""
        pass

    def test_transient_2d(self):
        """test Transient2D."""
        pass


if __name__ == "__main__":
    with open("./tests/reports/report.txt", "w", encoding="utf8") as reporter:
        suit = unittest.TestSuite()
        suit.addTest(TestFvmEqs("test_diffusion_2d"))

        runner = unittest.TextTestRunner(stream=reporter, verbosity=2)
        runner.run(suit)
