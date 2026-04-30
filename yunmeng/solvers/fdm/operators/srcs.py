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
from yunmeng.numerics.mesh import Grid, ElementType
from yunmeng.numerics.algos.topos import MeshTopo
from yunmeng.numerics.fields import DataHub, Field, Var
from typing import Callable


class Src01(IOperator):
    """Explicit source term operator on structured grids."""

    @classmethod
    def get_type(cls) -> OperatorType:
        return OperatorType.SRC

    @classmethod
    def get_name(cls) -> str:
        return "src01"

    def __init__(self, tau: float, source_func: Callable):
        self._mesh: Grid = None
        self._topo: MeshTopo = None
        self._bcs = None
        self._var = ""
        self._tau = tau
        self._source_func = source_func

    def prepare(
        self,
        fields: list[str],
        mesh: Grid,
        bounds: dict[int, dict[str, IBoundaryCondition]],
    ):
        if not isinstance(mesh, Grid):
            raise ValueError("FDM op src01 only supports Grid.")
        if not mesh.orthogonal:
            raise ValueError("FDM op src01 requires orthogonal grids.")
        if len(fields) != 1:
            raise ValueError("FDM op src01 only supports one field.")

        self._mesh = mesh
        self._bcs = bounds
        self._var = fields[0]
        self._topo = self._mesh.get_topo_assistant()

    def run(self, sources: DataHub, timestep: float) -> Field:
        old_field = sources.field(self._var, loc=ElementType.NODE).data
        new_field = old_field.copy()

        if len(old_field.mesh_shards) != 1:
            raise ValueError("FDM op src01 only supports cpu.")

        # Apply boundary conditions
        self._apply_bc(new_field)

        # Update internal nodes with source term
        new_field = self._update_internal(new_field, timestep)
        return new_field

    def _apply_bc(self, field: Field):
        """Apply boundary conditions to the field."""
        for nid in self._topo.boundary_nodes:
            bc = self._bcs[nid][self._var]
            value = bc.evaluate().value
            field[nid] = value

    def _update_internal(self, field: Field, dt: float) -> Field:
        """Update internal nodes with source term."""
        new_field = field.copy()

        for nid in self._topo.internal_nodes:
            coor = self._mesh.nodes[nid].coordinate
            source_val = self._tau * self._source_func(coor, field[nid])
            new_field[nid] = field[nid] + dt * source_val

        return new_field
