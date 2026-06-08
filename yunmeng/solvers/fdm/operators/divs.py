# -*- encoding: utf-8 -*-
"""
Copyright (C) 2025, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Divergence operators for the finite difference method.
"""

from yunmeng.solvers.interfaces import (
    IBoundaryCondition,
    IOperator,
    OperatorType,
    BoundaryType,
)
from yunmeng.numerics.mesh import Grid2D, ElementType
from yunmeng.numerics.algos.topos import MeshTopo
from yunmeng.numerics.fields import DataHub, Field, Variable, VariableType


class Div01(IOperator):
    """
    Divergence operator based on gradient result.
    """

    @classmethod
    def get_type(cls) -> OperatorType:
        return OperatorType.DIV

    @classmethod
    def get_name(cls) -> str:
        return "div01"

    def __init__(self, fields: list[str]):
        if len(fields) != 1:
            raise ValueError("FDM op div01 only supports one field.")
        self._var = fields[0]
        self._mesh: Grid2D = None

    @property
    def target_fields(self) -> list[str]:
        return [self._var]

    def prepare(
        self,
        mesh: Grid2D,
        bounds: dict[int, dict[str, IBoundaryCondition]] = None,
    ):
        if not isinstance(mesh, Grid2D):
            raise ValueError("FDM op grad02 only supports Grid2D.")
        self._mesh = mesh

    def run(self, sources: DataHub, dt: float = None) -> Field:
        """Calculate the divergence of the field."""
        if not isinstance(sources, DataHub):
            raise ValueError("FDM op div01 only supports DataHub.")

        grads = sources.grad(self._var, loc=ElementType.NODE).data
        if len(grads.mesh_shards) != 1:
            # TODO: Support multi-gpu operator
            raise ValueError("FDM op div01 only supports cpu.")

        divs = Field(grads.mesh_shards, VariableType.SCALAR, ElementType.NODE)
        for i in range(self._mesh.node_count):
            grad = grads[i]
            if grads.vtype == VariableType.TENSOR:
                div = Variable.scalar(grad[0, 0] + grad[1, 1] + grad[2, 2])
            else:
                div = Variable.scalar(grad[0] + grad[1] + grad[2])
            divs[i] = div
        return divs
