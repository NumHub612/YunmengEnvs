# -*- encoding: utf-8 -*-
from core.visuals.plotter import plot_field, plot_mesh
from core.visuals.plotter import (
    plot_mesh_geometry,
    plot_mesh_cloudmap,
    plot_mesh_scatters,
    plot_mesh_streamplot,
)
from core.numerics.mesh import Coordinate, Grid1D, Grid2D, Grid3D
from core.numerics.fields import Field, NodeField
import unittest
import numpy as np
import os
import shutil


class TestPlotters(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        print(f"\n---------- Testing < {cls.__name__} > \n")

    @classmethod
    def tearDownClass(cls):
        print("\n---------- Done \n")

    def setUp(self):
        self.is_show = True if __name__ == "__main__" else False
        self.save_dir = "./tests/results/"
        if os.path.exists(self.save_dir):
            shutil.rmtree(self.save_dir)
        os.makedirs(self.save_dir)

    def tearDown(self):
        if os.path.exists("./tests/results/"):
            shutil.rmtree("./tests/results/")
        os.makedirs("./tests/results/")

    def test_mesh(self):
        """Test plotting mesh"""
        # set 2d mesh
        low_left, upper_right = Coordinate(0, 0), Coordinate(2, 2)
        nx, ny = 41, 41
        grid2d = Grid2D(low_left, upper_right, nx, ny)

        for node in grid2d.nodes:
            coor = node.coordinate
            coor.z = np.sin(coor.x) * np.cos(coor.y)
            node.coordinate = coor

        plot_mesh(
            grid2d,
            title="2D Mesh",
            save_dir=self.save_dir,
            show=self.is_show,
            slice_set={"style": "slice_along_axis", "n": 6, "axis": "y"},
        )

        # set 3d mesh
        low_left, upper_right = Coordinate(0, 0, 0), Coordinate(2, 2, 2)
        nx, ny, nz = 11, 11, 11
        grid3d = Grid3D(low_left, upper_right, nx, ny, nz)

        plot_mesh(
            grid3d,
            title="3D Mesh",
            save_dir=self.save_dir,
            show=self.is_show,
            slice_set={"style": "slice_orthogonal"},
        )

    def test_grid1d(self):
        """Test plotting fields for 1D grid"""
        # set 1d mesh
        start, end = Coordinate(0), Coordinate(2 * np.pi)
        grid = Grid1D(start, end, 401)

        save_dir = os.path.join(self.save_dir, "grid1d")

        # set node scalar field
        scalar_values_n = np.array([np.random.rand(1) for i in range(grid.node_count)])
        scalar_field = Field.from_np(scalar_values_n, "node")

        plot_field(
            scalar_field,
            grid,
            title="1D Scalar Field",
            label="value",
            save_dir=save_dir,
            show=self.is_show,
            color="r",
            marker="o",
        )

        # set node vector field
        vector_values_n = np.random.rand(grid.node_count, 3)
        vector_field = Field.from_np(vector_values_n, "node")

        plot_field(
            vector_field,
            grid,
            title="1D Vector Field",
            label="u",
            save_dir=save_dir,
            show=self.is_show,
            dimension="x",
            color="r",
            marker="o",
        )

    def test_grid2d(self):
        """Test plotting fields for 2D grid"""
        # set 2d mesh
        low_left, upper_right = Coordinate(0, 0), Coordinate(2, 2)
        nx, ny = 41, 41
        grid = Grid2D(low_left, upper_right, nx, ny)

        save_dir = os.path.join(self.save_dir, "grid2d")

        # set node scalar field
        scalar_values_n = np.array([np.random.rand(1) for i in range(grid.node_count)])
        scaler_field_n = Field.from_np(scalar_values_n, "node")

        plot_field(
            scaler_field_n,
            grid,
            title="Node-Scalar-cloudmap",
            label="value",
            style="cloudmap",
            save_dir=save_dir,
            show=self.is_show,
        )

        plot_field(
            scaler_field_n,
            grid,
            title="Node-Scalar-scatter",
            label="value",
            style="scatter",
            save_dir=save_dir,
            show=self.is_show,
        )

        # set node vector field
        vector_values_n = np.random.rand(grid.node_count, 3)
        vector_values_n[:, 2] = 0.0
        vector_field_n = Field.from_np(vector_values_n, "node")

        plot_field(
            vector_field_n,
            grid,
            title="Node-Vector-cloudmap-x",
            label="u",
            style="cloudmap",
            save_dir=save_dir,
            show=self.is_show,
            dimension="x",
        )

        plot_field(
            vector_field_n,
            grid,
            title="Node-Vector-scatter-y",
            label="v",
            style="scatter",
            save_dir=save_dir,
            show=self.is_show,
            dimension="y",
        )

        plot_field(
            vector_field_n,
            grid,
            title="Node-Vector-streamplot",
            label="value",
            style="streamplot",
            save_dir=save_dir,
            show=self.is_show,
            color="b",
            mag=0.1,
        )

        # set cell scalar field
        scalar_values_c = np.array([np.random.rand(1) for i in range(grid.cell_count)])
        scaler_field_c = Field.from_np(scalar_values_c, "cell", "u")

        plot_field(
            scaler_field_c,
            grid,
            title="Cell-Scalar-Field",
            label="value",
            style="cloudmap",
            save_dir=save_dir,
            show=self.is_show,
        )

        # set cell vector field
        vector_values_c = np.random.rand(grid.cell_count, 3)
        vector_values_c[:, 2] = 0.0
        vector_field_c = Field.from_np(vector_values_c, "cell")

        plot_field(
            vector_field_c,
            grid,
            title="Cell-Vector-Field",
            label="u",
            style="cloudmap",
            save_dir=save_dir,
            show=self.is_show,
            dimension="x",
        )

    def test_grid3d(self):
        """Test plotting fields for 3D grid"""
        # set 3d mesh
        low_left, upper_right = Coordinate(0, 0, 0), Coordinate(2, 2, 2)
        nx, ny, nz = 11, 11, 11
        grid = Grid3D(low_left, upper_right, nx, ny, nz)

        save_dir = os.path.join(self.save_dir, "grid3d")

        # set scalar field
        scalar_values_n = np.array([[i % 100] for i in range(grid.node_count)])
        scalar_field = Field.from_np(scalar_values_n, "node", "u")

        plot_field(
            scalar_field,
            grid,
            title="scalar-cloudmap",
            label="value",
            save_dir=save_dir,
            show=self.is_show,
            slice_set={"style": "slice_along_axis", "n": 6, "axis": "y"},
        )

        plot_field(
            scalar_field,
            grid,
            title="scalar-scatter",
            label="value",
            style="scatter",
            save_dir=save_dir,
            show=self.is_show,
        )

        # set vector field
        vector_values_n = np.random.rand(grid.node_count, 3)
        vector_field = NodeField.from_np(vector_values_n, variable="u")

        plot_field(
            vector_field,
            grid,
            title="vector-cloudmap",
            label="u",
            style="cloudmap",
            save_dir=save_dir,
            show=self.is_show,
            dimension="x",
            slice_set={"style": "slice", "normal": [1, 1, 0]},
        )


if __name__ == "__main__":
    with open("./tests/reports/report.txt", "w", encoding="utf8") as reporter:
        # suit = unittest.TestLoader().loadTestsFromTestCase(TestPlotters)

        suit = unittest.TestSuite()
        suit.addTest(TestPlotters("test_grid3d"))

        runner = unittest.TextTestRunner(stream=reporter, verbosity=2)
        runner.run(suit)
