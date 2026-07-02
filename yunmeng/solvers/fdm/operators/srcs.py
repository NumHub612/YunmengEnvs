# -*- encoding: utf-8 -*-
"""
Copyright (C) 2025, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Source term operators for the finite difference method.
"""

from yunmeng.solvers.interfaces import (
    IBoundaryCondition,
    IOperator,
    OperatorType,
    BoundaryType,
)
from yunmeng.numerics.grids import Grid, ElementType
from yunmeng.numerics.algos import MeshTopo
from yunmeng.numerics.fields import DataHub, Field, VariableType
from typing import Callable


class Src01(IOperator):
    """Explicit source term operator on structured grids."""

    @classmethod
    def get_type(cls) -> OperatorType:
        return OperatorType.SRC

    @classmethod
    def get_name(cls) -> str:
        return "src01"

    def __init__(self, fields: list[str], tau: float, source_func: Callable):
        if len(fields) != 1:
            raise ValueError("FDM op src01 only supports one field.")

        self._mesh: Grid = None
        self._topo: MeshTopo = None
        self._bcs = None
        self._var = fields[0]
        self._tau = tau
        self._source_func = source_func

    @property
    def target_fields(self) -> list[str]:
        return [self._var]

    def prepare(
        self,
        mesh: Grid,
        bounds: dict[int, dict[str, IBoundaryCondition]],
    ):
        if not isinstance(mesh, Grid):
            raise ValueError("FDM op src01 only supports Grid.")
        if not mesh.uniform:
            raise ValueError("FDM op src01 requires uniform grids.")

        self._mesh = mesh
        self._bcs = bounds
        self._topo = self._mesh.get_topo_assistant()

    def run(self, sources: Field | DataHub, dt: float = None) -> Field:
        if isinstance(sources, Field):
            old_field = sources
        else:
            old_field = sources.field(self._var, loc=ElementType.NODE).data
        new_field = Field(old_field.mesh_shards, old_field.vtype, old_field.etype)

        if len(old_field.mesh_shards) != 1:
            raise ValueError("FDM op src01 only supports cpu.")

        for nid in range(self._mesh.node_count):
            coor = self._mesh.nodes[nid].coordinate
            source_val = self._tau * self._source_func(coor, old_field[nid])
            new_field[nid] = source_val

        return new_field

    def _apply_bc(self, field: Field):
        """Apply boundary conditions to the field."""
        for nid in self._topo.boundary_nodes:
            bc = self._bcs[nid][self._var]
            value = bc.evaluate().value
            field[nid] = value
