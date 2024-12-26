# -*- encoding: utf-8 -*-
from core.numerics.mesh import Coordinate, Grid1D, Grid2D, Grid3D
import unittest


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
            grid.retrieve_node_neighborhoods(0), [1, None, 4, None, None, None]
        )
        self.assertEqual(
            grid.retrieve_node_neighborhoods(1), [2, 0, 5, None, None, None]
        )
        self.assertEqual(
            grid.retrieve_node_neighborhoods(3), [None, 2, 7, None, None, None]
        )
        self.assertEqual(grid.retrieve_node_neighborhoods(6), [7, 5, 10, 2, None, None])
        self.assertEqual(
            grid.retrieve_node_neighborhoods(14), [15, 13, None, 10, None, None]
        )

        self.assertEqual(grid.match_cell(1, 1), 4)
        self.assertEqual(grid.match_cell(2, 2), 8)
        self.assertEqual(grid.match_cell(1, 2), 5)

        self.assertEqual(
            grid.retrieve_cell_neighborhoods(0), [1, None, 3, None, None, None]
        )
        self.assertEqual(
            grid.retrieve_cell_neighborhoods(2), [None, 1, 5, None, None, None]
        )
        self.assertEqual(grid.retrieve_cell_neighborhoods(4), [5, 3, 7, 1, None, None])
        self.assertEqual(
            grid.retrieve_cell_neighborhoods(8), [None, 7, None, 5, None, None]
        )

    def test_grid3d(self):
        """test Grid3D"""
        return True


if __name__ == "__main__":
    with open("./tests/reports/report.txt", "w", encoding="utf8") as reporter:
        suit = unittest.TestSuite()
        suit.addTest(TestGrids("test_grid2d"))

        runner = unittest.TextTestRunner(stream=reporter, verbosity=2)
        runner.run(suit)
