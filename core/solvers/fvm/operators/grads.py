# -*- encoding: utf-8 -*-
"""
Copyright (C) 2025, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Grad operators for the finite volume method.
"""
from core.solvers.interfaces import IOperator, OperatorType
from core.numerics.matrix import LinearEqs
from core.numerics.fields import Field, NodeField, Tensor, Vector, VariableType
from core.numerics.mesh import Grid, MeshTopo, MeshGeom, ElementType

import numpy as np


class Grad01(IOperator):
    """
    Simple implicit backward gradient operator excuted on `Grid` mesh in fvm.

    scheme:
        - implicit method.

    limits:
        - only supports `Grid` mesh.
        - only supports scalar and vector fields.
        - only supports cell centered fields.
    """

    @classmethod
    def get_type(self) -> OperatorType:
        return OperatorType.GRAD

    @classmethod
    def get_name(cls) -> str:
        return "grad01"

    def __init__(self):
        self._mesh = None
        self._topo = None
        self._geom = None

    def prepare(self, mesh: Grid, **kwargs):
        if not isinstance(mesh, Grid):
            raise ValueError("Fvm Grad01 operator only supports Grid.")

        self._mesh = mesh
        self._topo = self._mesh.get_topo_assistant()
        self._geom = self._mesh.get_geom_assistant()

    def run(self, source: Field) -> Field | LinearEqs:
        src_type = source.dtype
        if src_type != VariableType.SCALAR and src_type != VariableType.VECTOR:
            raise ValueError(
                "Fvm Grad01 operator only supports scalar and vector fields."
            )
        if source.etype != ElementType.CELL:
            raise ValueError("Fvm Grad01 operator only supports cell fields.")

        if src_type == VariableType.SCALAR:
            grand_func = self._calculate_scalar_grad
        else:
            grand_func = self._calculate_vector_grad

        grads = np.zeros((source.size, 3))
        for cid in range(self._mesh.cell_count):
            grads[cid] = grand_func(source, cid)

        return NodeField.from_data(grads)

    def _calculate_scalar_grad(self, source, element) -> Vector:
        """Calculate the gradient of scalar."""
        face_values = []
        for nb in self._mesh.retrieve_cell_neighbours(element):
            # east, west, north, south, top, bot
            if nb is not None:
                val = (source[element] + source[nb]) / 2.0
                face_values.append(val)
            else:
                face_values.append(None)

        # TODO: get face by cells index
        dists = [self._mesh.dx, self._mesh.dy, self._mesh.dz]
        results = []
        for i in range(0, 6, 2):
            dist = dists[i // 2]
            if face_values[i] is not None and face_values[i + 1] is not None:
                grad = (face_values[i + 1] - face_values[i]) / dist
                results.append(grad.value)
            else:
                results.append(0.0)  # TODO: can be better
        return results

    def _calculate_vector_grad(self, source, element, neighbours) -> Tensor:
        """Calculate the gradient of vector."""
        pass
