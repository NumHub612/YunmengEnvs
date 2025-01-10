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
    ) -> Scalar:
        """Excute laplacian operator on scalar field."""
        if element in self._topo.boundary_nodes_indexes:
            return Scalar.zero()

        east, west, north, south, top, bot = neighbours
        results = []

        for indexes in [(east, west), (north, south), (top, bot)]:
            forward, backward = indexes
            if forward is None:
                results.append(0.0)
            else:
                ds1 = self._geom.calucate_node_to_node_distance(element, forward)
                part1 = (self._field[forward] - self._field[element]) / ds1

                ds2 = self._geom.calucate_node_to_node_distance(element, backward)
                part2 = (self._field[element] - self._field[backward]) / ds2

                ds = 0.5 * (ds1 + ds2)
                results.append((part1 - part2) / ds)

        return Scalar(sum(results))

    def _calculate_vector_laplacian(
        self,
        element: int,
        neighbours: list[int],
    ) -> Vector:
        """Excute laplacian operator on vector field."""
        if element in self._topo.boundary_nodes_indexes:
            return Vector.zero()

        east, west, north, south, top, bot = neighbours
        dists, values = [], []
        for indexes in [(east, west), (north, south), (top, bot)]:
            forward, backward = indexes
            if forward is None:
                dists.append((None, None, None))
                values.append((None, None))
            else:
                ds1 = self._geom.calucate_node_to_node_distance(element, forward)
                ds2 = self._geom.calucate_node_to_node_distance(element, backward)
                ds = 0.5 * (ds1 + ds2)
                dists.append((ds1, ds2, ds))

                values.append(
                    (self._field[forward].to_np(), self._field[backward].to_np())
                )

        elem_value = self._field[element].to_np()
        results = []
        for i in range(3):
            result = []
            for dist, value in zip(dists, values):
                ds1, ds2, ds = dist
                if ds1 is None:
                    result.append(0.0)
                else:
                    forward, backward = value
                    part1 = (forward[i] - elem_value[i]) / ds1
                    part2 = (elem_value[i] - backward[i]) / ds2
                    result.append((part1 - part2) / ds)

            results.append(sum(result))

        return Vector(*results)
