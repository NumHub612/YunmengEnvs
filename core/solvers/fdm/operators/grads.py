# -*- encoding: utf-8 -*-
"""
Copyright (C) 2025, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Grad operators for the finite difference method.
"""
from core.solvers.interfaces.IEquation import IOperator
from core.numerics.matrix import LinearEqs
from core.numerics.fields import Field, NodeField, Tensor, Vector
from core.numerics.mesh import Grid

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
        self._source = None
        self._mesh = None
        self._topo = None
        self._geom = None

    @property
    def type(self) -> str:
        return "GRAD"

    def prepare(self, mesh: Grid, **kwargs):
        if not isinstance(mesh, Grid):
            raise ValueError("Grad01 operator only supports Grid.")

        self._mesh = mesh
        self._topo = self._mesh.get_topo_assistant()
        self._geom = self._mesh.get_geom_assistant()
        self._source = None

    def run(self, source: Field) -> Field | LinearEqs:
        src_field_type = source.dtype
        if src_field_type not in ["scalar", "vector"]:
            raise ValueError("Grad01 operator only supports scalar and vector fields.")
        self._source = source

        if src_field_type == "scalar":
            results = NodeField(
                source.size,
                "vector",
                data=Vector.zero(),
                variable=source.variable,
            )
        else:
            results = NodeField(
                source.size,
                "tensor",
                data=Tensor.zero(),
                variable=source.variable,
            )

        for element in self._mesh.node_indices:
            neighbours = self._mesh.retrieve_node_neighborhoods(element)

            # calculate
            if src_field_type == "scalar":
                grad = self._calculate_scalar_grad(element, neighbours)
            else:
                grad = self._calculate_vector_grad(element, neighbours)

            results[element] = grad

        return results

    def _calculate_scalar_grad(
        self,
        element: int,
        neighbours: list[int],
    ) -> Vector:
        """Calculate the gradient of scalar."""
        if element in self._topo.boundary_nodes_indices:
            return Vector.zero()

        east, west, north, south, top, bot = neighbours
        results = []

        for nb in [west, south, bot]:
            if nb is not None:
                ds = self._geom.calucate_node_to_node_distance(element, nb)
                grad = (self._source[element] - self._source[nb]) / ds
                results.append(grad)
            else:
                results.append(0.0)

        return Vector(*results)

    def _calculate_vector_grad(
        self,
        element: int,
        neighbours: list[int],
    ) -> Tensor:
        """Calculate the gradient of vector."""
        if element in self._topo.boundary_nodes_indices:
            return Tensor.zero()

        east, west, north, south, top, bot = neighbours
        results = []

        ds, vecs = [], []
        for nb in [west, south, bot]:
            if nb is not None:
                ds.append(self._geom.calucate_node_to_node_distance(element, nb))
                vecs.append(self._source[nb].to_np())
            else:
                ds.append(None)
                vecs.append(None)

        elem_vec = self._source[element].to_np()
        for i in range(3):
            row = []
            for vec, d in zip(vecs, ds):
                if d is not None:
                    row.append((elem_vec[i] - vec[i]) / d)
                else:
                    row.append(0.0)
            results.append(row)

        results = np.array(results).T
        return Tensor.from_np(results)
