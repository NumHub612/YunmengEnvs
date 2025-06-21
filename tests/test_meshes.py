# -*- encoding: utf-8 -*-
from core.numerics.mesh import (
    Coordinate,
    GenericMesh,
    ElementType,
    MeshGeom,
    MeshTopo,
)
import numpy as np
import unittest
import tempfile
import shutil


class TestGenericMesh(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print(f"\n---------- Testing < {cls.__name__} > \n")

    @classmethod
    def tearDownClass(cls):
        print("\n---------- Done \n")

    def setUp(self):
        # 构造一个3x3节点的2x2单元的正方形网格
        # 节点编号:
        # 6---7---8
        # |   |   |
        # 3---4---5
        # |   |   |
        # 0---1---2
        self.nodes = np.array(
            [
                [0, 0, 0],
                [1, 0, 0],
                [2, 0, 0],
                [0, 1, 0],
                [1, 1, 0],
                [2, 1, 0],
                [0, 2, 0],
                [1, 2, 0],
                [2, 2, 0],
            ]
        )
        # 每个小方格单元的边（face）按节点对定义
        self.faces = np.array(
            [
                [0, 1],
                [1, 2],  # 底边
                [3, 4],
                [4, 5],  # 中下边
                [6, 7],
                [7, 8],  # 顶边
                [0, 3],
                [3, 6],  # 左边
                [1, 4],
                [4, 7],  # 中竖边
                [2, 5],
                [5, 8],  # 右边
                [1, 3],
                [2, 4],
                [4, 6],
                [5, 7],  # 内部对角线（可选）
            ]
        )
        # 四个单元，每个单元由4条边组成（按face索引）
        self.cells = np.array(
            [
                [0, 7, 2, 8],  # 左下
                [1, 8, 3, 10],  # 右下
                [2, 7, 4, 9],  # 左上
                [3, 9, 5, 11],  # 右上
            ]
        )
        self.mesh = GenericMesh(self.nodes, self.faces, self.cells)

    def test_basic_properties(self):
        self.assertEqual(self.mesh.node_count, 9)
        self.assertEqual(self.mesh.face_count, len(self.faces))
        self.assertEqual(self.mesh.cell_count, 4)
        self.assertEqual(len(self.mesh.nodes), 9)
        self.assertEqual(len(self.mesh.faces), len(self.faces))
        self.assertEqual(len(self.mesh.cells), 4)

    def test_get_methods(self):
        nodes = self.mesh.get_nodes([0, 4, 8])
        self.assertEqual(len(nodes), 3)
        faces = self.mesh.get_faces([0, 5, 10])
        self.assertEqual(len(faces), 3)
        cells = self.mesh.get_cells([0, 3])
        self.assertEqual(len(cells), 2)

    def test_group_methods(self):
        self.mesh.set_group(ElementType.NODE, "corners", [0, 2, 6, 8])
        indices, etype = self.mesh.get_group("corners")
        self.assertEqual(sorted(indices), [0, 2, 6, 8])
        self.assertEqual(etype, ElementType.NODE)
        self.mesh.delete_group("corners")
        with self.assertRaises(ValueError):
            self.mesh.get_group("corners")

    def test_save_and_load(self):
        tmpdir = tempfile.mkdtemp()
        try:
            self.mesh.save(tmpdir)
            loaded = GenericMesh.load(tmpdir)
            self.assertEqual(loaded.node_count, 9)
            self.assertEqual(loaded.face_count, len(self.faces))
            self.assertEqual(loaded.cell_count, 4)
        finally:
            shutil.rmtree(tmpdir)


class TestMeshTopo(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print(f"\n---------- Testing < {cls.__name__} > \n")

    @classmethod
    def tearDownClass(cls):
        print("\n---------- Done \n")

    def setUp(self):
        nodes = np.array(
            [
                [0, 0, 0],
                [1, 0, 0],
                [2, 0, 0],
                [0, 1, 0],
                [1, 1, 0],
                [2, 1, 0],
                [0, 2, 0],
                [1, 2, 0],
                [2, 2, 0],
            ]
        )
        faces = np.array(
            [
                [0, 1],
                [1, 2],
                [3, 4],
                [4, 5],
                [6, 7],
                [7, 8],
                [0, 3],
                [3, 6],
                [1, 4],
                [4, 7],
                [2, 5],
                [5, 8],
                [1, 3],
                [2, 4],
                [4, 6],
                [5, 7],
            ]
        )
        cells = np.array(
            [
                [0, 7, 2, 8],
                [1, 8, 3, 10],
                [2, 7, 4, 9],
                [3, 9, 5, 11],
            ]
        )
        self.mesh = GenericMesh(nodes, faces, cells)
        self.topo = MeshTopo(self.mesh)

    def test_boundary_and_interior(self):
        # 角点和边界点
        boundary_nodes = set(self.topo.boundary_nodes)
        self.assertTrue(0 in boundary_nodes and 8 in boundary_nodes)
        # 内部点
        self.assertTrue(4 in self.topo.interior_nodes)
        # 边界面
        boundary_faces = set(self.topo.boundary_faces)
        self.assertTrue(0 in boundary_faces and 5 in boundary_faces)
        # 内部面
        self.assertTrue(
            any(f not in boundary_faces for f in range(self.mesh.face_count))
        )

    def test_connectivity(self):
        fc = self.topo.face_cells
        self.assertTrue(isinstance(fc, dict))
        nf = self.topo.node_faces
        self.assertTrue(isinstance(nf, dict))
        nc = self.topo.node_cells
        self.assertTrue(isinstance(nc, dict))
        cn = self.topo.cell_nodes
        self.assertTrue(isinstance(cn, dict))
        cnb = self.topo.cell_neighbours
        self.assertTrue(isinstance(cnb, dict))
        nnb = self.topo.node_neighbours
        self.assertTrue(isinstance(nnb, dict))

    def test_indices(self):
        self.assertEqual(self.topo.face_id_indices[0], 0)
        self.assertEqual(self.topo.node_id_indices[1], 1)
        self.assertEqual(self.topo.cell_id_indices[0], 0)


class TestMeshGeom(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print(f"\n---------- Testing < {cls.__name__} > \n")

    @classmethod
    def tearDownClass(cls):
        print("\n---------- Done \n")

    def setUp(self):
        nodes = np.array(
            [
                [0, 0, 0],
                [1, 0, 0],
                [2, 0, 0],
                [0, 1, 0],
                [1, 1, 0],
                [2, 1, 0],
                [0, 2, 0],
                [1, 2, 0],
                [2, 2, 0],
            ]
        )
        faces = np.array(
            [
                [0, 1],
                [1, 2],
                [3, 4],
                [4, 5],
                [6, 7],
                [7, 8],
                [0, 3],
                [3, 6],
                [1, 4],
                [4, 7],
                [2, 5],
                [5, 8],
                [1, 3],
                [2, 4],
                [4, 6],
                [5, 7],
            ]
        )
        cells = np.array(
            [
                [0, 7, 2, 8],
                [1, 8, 3, 10],
                [2, 7, 4, 9],
                [3, 9, 5, 11],
            ]
        )
        self.mesh = GenericMesh(nodes, faces, cells)
        self.geom = MeshGeom(self.mesh)

    def test_static_methods(self):
        c1 = Coordinate(0, 0, 0)
        c2 = Coordinate(1, 0, 0)
        dist = MeshGeom.calculate_distance(c1, c2)
        self.assertAlmostEqual(dist, 1.0)
        center = MeshGeom.calculate_center([c1, c2])
        self.assertAlmostEqual(center.x, 0.5)
        self.assertAlmostEqual(center.y, 0.0)

    def test_face_areas(self):
        areas = self.geom.face_areas
        self.assertEqual(len(areas), self.mesh.face_count)
        for a in areas:
            self.assertGreaterEqual(a, 0.0)

    def test_face_perimeters(self):
        perims = self.geom.face_perimeters
        self.assertEqual(len(perims), self.mesh.face_count)

    def test_cell_volumes(self):
        vols = self.geom.cell_volumes
        self.assertEqual(len(vols), self.mesh.cell_count)
        for v in vols:
            self.assertGreaterEqual(v, 0.0)


if __name__ == "__main__":
    with open("./tests/reports/report.txt", "w", encoding="utf8") as reporter:
        runner = unittest.TextTestRunner(stream=reporter, verbosity=2)

        suit = unittest.TestLoader().loadTestsFromTestCase(TestGenericMesh)
        runner.run(suit)
        suit = unittest.TestLoader().loadTestsFromTestCase(TestMeshTopo)
        runner.run(suit)
        suit = unittest.TestLoader().loadTestsFromTestCase(TestMeshGeom)
        runner.run(suit)
