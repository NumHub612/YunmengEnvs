# -*- encoding: utf-8 -*-
from core.numerics.mesh import Coordinate, GenericMesh, AdaptiveRectangularMesh
import unittest


class TestMeshes(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        print(f"\n---------- Testing < {cls.__name__} > \n")

    @classmethod
    def tearDownClass(cls):
        print("\n---------- Done \n")

    def setUp(self):
        from core.numerics.mesh import Grid2D

        low_left, upper_right = Coordinate(0, 0), Coordinate(100, 100)
        nx, ny = 5, 5
        grid = Grid2D(low_left, upper_right, nx, ny)

        self.nodes = [node.coordinate.to_np() for node in grid.nodes]
        self.faces = [face.nodes for face in grid.faces]
        self.cells = [cell.faces for cell in grid.cells]

    def tearDown(self):
        pass

    def test_generic_mesh(self):
        """Test the generic mesh."""
        mesh = GenericMesh(self.nodes, self.faces, self.cells)

    def test_adaptive_mesh(self):
        """Test the adaptive rectangular mesh."""
        mesh = AdaptiveRectangularMesh(self.nodes, self.faces, self.cells)
        self.assertEqual(mesh.dimension, "2d")
        self.assertEqual(mesh.version, 1)

        # mesh.refine_cells([0, 1, 2])
        # self.assertEqual(mesh.version, 2)

        # mesh.relax_cells([0, 1, 2])
        # self.assertEqual(mesh.version, 3)


if __name__ == "__main__":
    with open("./tests/reports/report.txt", "w", encoding="utf8") as reporter:
        suit = unittest.TestSuite()
        suit.addTest(TestMeshes("test_adaptive_mesh"))

        runner = unittest.TextTestRunner(stream=reporter, verbosity=2)
        runner.run(suit)
