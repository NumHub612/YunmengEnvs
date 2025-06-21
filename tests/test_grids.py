# -*- encoding: utf-8 -*-
from core.numerics.mesh import Coordinate, Grid1D, Grid2D, Grid3D
import unittest
import numpy as np


class TestGrids(unittest.TestCase):

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

    def test_grid1d(self):
        """test Grid1D"""
        start, end = Coordinate(0), Coordinate(2 * np.pi)
        grid = Grid1D(start, end, 401)
        return True

    def test_grid2d(self):
        """test Grid2D"""
        low_left, upper_right = Coordinate(0, 0), Coordinate(3, 3)
        nx, ny = 4, 4
        grid = Grid2D(low_left, upper_right, nx, ny)

        self.assertEqual(grid.cell_count, 9)
        self.assertEqual(grid.face_count, 24)
        self.assertEqual(grid.node_count, 16)

        self.assertEqual(grid.match_node(1, 1), 5)
        self.assertEqual(grid.match_node(2, 2), 10)
        self.assertEqual(grid.match_node(3, 2), 14)

        self.assertEqual(
            grid.retrieve_node_neighbours(0), [4, None, 1, None, None, None]
        )
        self.assertEqual(grid.retrieve_node_neighbours(1), [5, None, 2, 0, None, None])
        self.assertEqual(
            grid.retrieve_node_neighbours(3), [7, None, None, 2, None, None]
        )
        self.assertEqual(grid.retrieve_node_neighbours(6), [10, 2, 7, 5, None, None])
        self.assertEqual(
            grid.retrieve_node_neighbours(14), [None, 10, 15, 13, None, None]
        )

        self.assertEqual(grid.match_cell(1, 1), 4)
        self.assertEqual(grid.match_cell(2, 2), 8)
        self.assertEqual(grid.match_cell(1, 2), 5)

        self.assertEqual(
            grid.retrieve_cell_neighbours(0), [3, None, 1, None, None, None]
        )
        self.assertEqual(
            grid.retrieve_cell_neighbours(2), [5, None, None, 1, None, None]
        )
        self.assertEqual(grid.retrieve_cell_neighbours(4), [7, 1, 5, 3, None, None])
        self.assertEqual(
            grid.retrieve_cell_neighbours(8), [None, 5, None, 7, None, None]
        )

    def test_grid3d(self):
        """test Grid3D"""
        low_left, upper_right = Coordinate(0, 0, 0), Coordinate(2, 2, 2)
        nx, ny, nz = 3, 3, 3
        grid = Grid3D(low_left, upper_right, nx, ny, nz)

        self.assertEqual(grid.cell_count, 8)
        self.assertEqual(grid.face_count, 36)
        self.assertEqual(grid.node_count, 27)

        self.assertEqual(grid.match_node(1, 1, 0), 4)
        self.assertEqual(grid.match_node(1, 1, 1), 13)
        self.assertEqual(grid.match_node(1, 2, 1), 16)
        self.assertEqual(grid.match_node(2, 2, 2), 26)
        self.assertEqual(grid.match_node(2, 1, 2), 23)

        self.assertEqual(grid.retrieve_node_neighbours(2), [None, 1, 5, None, 11, None])
        self.assertEqual(grid.retrieve_node_neighbours(4), [5, 3, 7, 1, 13, None])
        self.assertEqual(grid.retrieve_node_neighbours(13), [14, 12, 16, 10, 22, 4])
        self.assertEqual(
            grid.retrieve_node_neighbours(25), [26, 24, None, 22, None, 16]
        )

        self.assertEqual(grid.match_cell(0, 1, 0), 2)
        self.assertEqual(grid.match_cell(1, 1, 0), 3)
        self.assertEqual(grid.match_cell(1, 0, 1), 5)
        self.assertEqual(grid.match_cell(0, 1, 1), 6)
        self.assertEqual(grid.match_cell(1, 1, 1), 7)

        self.assertEqual(grid.retrieve_cell_neighbours(1), [None, 0, 3, None, 5, None])
        self.assertEqual(grid.retrieve_cell_neighbours(6), [7, None, None, 4, None, 2])

        return True


if __name__ == "__main__":
    with open("./tests/reports/report.txt", "w", encoding="utf8") as reporter:
        suit = unittest.TestSuite()
        suit.addTest(TestGrids("test_grid2d"))
        suit.addTest(TestGrids("test_grid3d"))

        runner = unittest.TextTestRunner(stream=reporter, verbosity=2)
        runner.run(suit)
