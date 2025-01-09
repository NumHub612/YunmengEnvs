# -*- encoding: utf-8 -*-
"""
Copyright (C) 2025, The YunmengEnvs Contributors. Join us, share your ideas!  

Grad operators for the finite difference method.
"""
from core.solvers.interfaces.IEquation import IOperator
from core.numerics.matrix import LinearEqs
from core.numerics.fields import Field, Variable, Tensor, Vector
from core.numerics.mesh import Grid, MeshGeom, MeshTopo

import numpy as np


class Grad01(IOperator):
    """
    Simple implicit backward gradient operator excuted on `Grid` mesh in fdm.

    scheme:
        - implicit method
        - backward difference in space

    limits:
        - only supports `Grid` mesh.
        - only supports scalar and vector fields.
    """

    def __init__(self):
        self._field = None
        self._topo = None
        self._geom = None

    @property
    def type(self) -> str:
        return "GRAD"

    def prepare(self, field: Field, topo: MeshTopo, geom: MeshGeom, **kwargs):
        self._field = field
        self._topo = topo
        self._geom = geom

        if not isinstance(self._topo.get_mesh(), Grid):
            raise ValueError("Grad01 operator only supports Grid.")

        data_type = self._field.dtype
        if data_type not in ["scalar", "vector"]:
            raise ValueError("Grad01 operator only supports scalar and vector fields.")

    def run(self, element: int) -> Variable | LinearEqs:
        neighbours = self._topo.get_mesh().retrieve_node_neighborhoods(element)
        data_type = self._field.dtype

        # calculate
        if data_type == "scalar":
            result = self._calculate_scalar_grad(element, neighbours)
        else:
            result = self._calculate_vector_grad(element, neighbours)
        return result

    def _calculate_scalar_grad(
        self,
        element: int,
        neighbours: list[int],
    ) -> Vector:
        """Calculate the gradient of scalar."""
        if element in self._topo.boundary_nodes_indexes:
            return Vector.zero()

        east, west, north, south, top, bot = neighbours
        results = []

        dx = self._geom.calucate_node_to_node_distance(element, west)
        x_part = self._field[element] - self._field[west]
        results.append(x_part / dx)

        dy = self._geom.calucate_node_to_node_distance(element, south)
        y_part = self._field[element] - self._field[south]
        results.append(y_part / dy)

        dz = self._geom.calucate_node_to_node_distance(element, bot)
        z_part = self._field[element] - self._field[bot]
        results.append(z_part / dz)

        return Vector(*results)

    def _calculate_vector_grad(
        self,
        element: int,
        neighbours: list[int],
    ) -> Tensor:
        """Calculate the gradient of vector."""
        if element in self._topo.boundary_nodes_indexes:
            return Tensor.zero()

        east, west, north, south, top, bot = neighbours
        results = []

        dx = self._geom.calucate_node_to_node_distance(element, west)
        dy = self._geom.calucate_node_to_node_distance(element, south)
        dz = self._geom.calucate_node_to_node_distance(element, bot)

        elem_vec = self._field[element].to_np()
        west_vec = self._field[west].to_np()
        north_vec = self._field[south].to_np()
        bot_vec = self._field[bot].to_np()

        for i in range(3):
            row = []

            x_part = elem_vec[i] - west_vec[i]
            row.append(x_part / dx)

            y_part = elem_vec[i] - north_vec[i]
            row.append(y_part / dy)

            z_part = elem_vec[i] - bot_vec[i]
            row.append(z_part / dz)

            results.append(row)

        results = np.array(results).T
        return Tensor.from_np(results)
