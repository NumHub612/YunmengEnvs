# -*- encoding: utf-8 -*-
"""
Copyright (C) 2025, The YunmengEnvs Contributors. Join us, share your ideas!  

Laplacian operators for the finite difference method.
"""
from core.solvers.interfaces.IEquation import IOperator
from core.numerics.matrix import LinearEqs
from core.numerics.fields import Field, Variable, Vector, Scalar
from core.numerics.mesh import MeshGeom, MeshTopo, Grid

import numpy as np


class Lap01(IOperator):
    """
    Simple implicit second-order Laplacian operator on `Grid` mesh in fdm.

    scheme:
        - implicit method
        - central difference approximation

    limits:
        - only supports `Grid` mesh.
    """

    def __init__(self):
        self._field = None
        self._topo = None
        self._geom = None

    @property
    def type(self) -> str:
        return "LAPLACIAN"

    def prepare(self, field: Field, topo: MeshTopo, geom: MeshGeom, **kwargs):
        self._field = field
        self._topo = topo
        self._geom = geom

        if not isinstance(self._topo.get_mesh(), Grid):
            raise ValueError("Grad01 operator only supports Grid.")

        data_type = self._field.dtype
        if data_type not in ["scalar", "vector"]:
            raise ValueError("Lap01 operator only supports scalar and vector fields.")

    def run(self, element: int) -> Variable | LinearEqs:
        neighbours = self._topo.get_mesh().retrieve_node_neighborhoods(element)
        data_type = self._field.dtype

        # calculate
        if data_type == "scalar":
            result = self._calculate_scalar_laplacian(element, neighbours)
        else:
            result = self._calculate_vector_laplacian(element, neighbours)

        return result

    def _calculate_scalar_laplacian(
        self,
        element: int,
        neighbours: list[int],
    ) -> Vector:
        """Excute laplacian operator on scalar field."""
        if element in self._topo.boundary_nodes_indexes:
            return Scalar.zero()

        south, north, east, west, top, bot = neighbours
        results = []

        dx2 = self._geom.calucate_node_to_node_distance(element, west)
        x_part2 = self._field[element] - self._field[west]

        dx1 = self._geom.calucate_node_to_node_distance(element, east)
        x_part1 = self._field[east] - self._field[element]

        dx = 0.5 * (dx1 + dx2)
        x_part = x_part1 - x_part2
        results.append(x_part / dx)

        dy2 = self._geom.calucate_node_to_node_distance(element, south)
        y_part2 = self._field[element] - self._field[south]

        dy1 = self._geom.calucate_node_to_node_distance(element, north)
        y_part1 = self._field[north] - self._field[element]

        dy = 0.5 * (dy1 + dy2)
        y_part = y_part1 - y_part2
        results.append(y_part / dy)

        dz2 = self._geom.calucate_node_to_node_distance(element, bot)
        z_part2 = self._field[element] - self._field[bot]

        dz1 = self._geom.calucate_node_to_node_distance(element, top)
        z_part1 = self._field[top] - self._field[element]

        dz = 0.5 * (dz1 + dz2)
        z_part = z_part1 - z_part2
        results.append(z_part / dz)

        return Scalar(sum(results))

    def _calculate_vector_laplacian(
        self,
        element: int,
        neighbours: list[int],
    ) -> Vector:
        """Excute laplacian operator on vector field."""
        if element in self._topo.boundary_nodes_indexes:
            return Vector.zero()

        south, north, east, west, top, bot = neighbours
        results = []

        dx1 = self._geom.calucate_node_to_node_distance(element, east)
        dx2 = self._geom.calucate_node_to_node_distance(element, west)
        dx = 0.5 * (dx1 + dx2)

        dy1 = self._geom.calucate_node_to_node_distance(element, north)
        dy2 = self._geom.calucate_node_to_node_distance(element, south)
        dy = 0.5 * (dy1 + dy2)

        dz1 = self._geom.calucate_node_to_node_distance(element, top)
        dz2 = self._geom.calucate_node_to_node_distance(element, bot)
        dz = 0.5 * (dz1 + dz2)

        elem_vec = self._field[element].to_np()
        west_vec = self._field[west].to_np()
        east_vec = self._field[east].to_np()
        north_vec = self._field[north].to_np()
        south_vec = self._field[south].to_np()
        bot_vec = self._field[bot].to_np()
        top_vec = self._field[top].to_np()

        for i in range(3):
            x_part = (east_vec[i] - elem_vec[i]) / dx1 - (
                elem_vec[i] - west_vec[i]
            ) / dx2

            y_part = (north_vec[i] - elem_vec[i]) / dy1 - (
                elem_vec[i] - south_vec[i]
            ) / dy2

            z_part = (top_vec[i] - elem_vec[i]) / dz1 - (elem_vec[i] - bot_vec[i]) / dz2
            results.append(x_part / dx + y_part / dy + z_part / dz)

        return Vector(*results)
