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
from yunmeng.numerics.mesh import Grid, ElementType
from yunmeng.numerics.algos.topos import MeshTopo
from yunmeng.numerics.fields import DataHub, Field


class Div01(IOperator):
    """
    First order upwind explicit divergence operator on structured grids.
    """

    @classmethod
    def get_type(cls) -> OperatorType:
        return OperatorType.DIV

    @classmethod
    def get_name(cls) -> str:
        return "div01"

    def __init__(self):
        self._mesh: Grid = None
        self._topo: MeshTopo = None

        self._bcs = None
        self._var = ""
        self._dx = None
        self._dy = None

    def prepare(
        self,
        fields: list[str],
        mesh: Grid,
        bounds: dict[int, dict[str, IBoundaryCondition]],
    ):
        if not isinstance(mesh, Grid):
            raise ValueError("FDM op div01 only supports Grid.")
        if not mesh.orthogonal:
            # TODO: Support non-orthogonal grids
            raise ValueError("FDM op div01 requires orthogonal grids.")
        if len(fields) != 1:
            raise ValueError("FDM op div01 only supports one field.")
        for bc in bounds.values():
            # TODO: Support more types of boundary conditions
            for v in bc.values():
                if v.get_type() != BoundaryType.VALUE:
                    raise ValueError("FDM op div01 requires value BC.")

        self._mesh = mesh
        self._bcs = bounds
        self._var = fields[0]

        self._topo = self._mesh.get_topo_assistant()
        self._dx = self._mesh.lx / (self._mesh.nx - 1)
        self._dy = self._mesh.ly / (self._mesh.ny - 1)

    def run(self, sources: DataHub, timestep: float) -> Field:
        old_field = sources.field(self._var, loc=ElementType.NODE).data
        new_field = Field.from_field(old_field)
        if len(old_field.mesh_shards) != 1:
            # TODO: Support multi-gpu divergence operator
            raise ValueError("FDM op div01 only supports cpu.")

        # Apply boundary conditions
        self._apply_bc(new_field)

        # Update internal nodes
        new_field = self._update_internal(new_field, timestep)

        return new_field

    def _apply_bc(self, field: Field):
        """Apply boundary conditions to the field."""
        for nid in self._topo.boundary_nodes:
            bc = self._bcs[nid][self._var]
            value = bc.evaluate().value
            field[nid] = value

    def _update_internal(self, field: Field, dt: float) -> Field:
        """Update the internal nodes of the field."""
        # Upwind check
        F = lambda c: (max(c / (abs(c) + 1e-6), 0), max(-c / (abs(c) + 1e-6), 0))

        new_field = Field.from_field(field)
        kx = dt / self._dx
        ky = dt / self._dy

        for nid in self._topo.internal_nodes:
            # Neighbour nodes
            e, w, n, s, _, _ = self._mesh.get_node_neighbours(nid)
            ui = field[nid]

            # Horizontal flux
            fh1, fh2 = F(ui[0])
            ue = ui * fh1 + field[e] * fh2
            uw = field[w] * fh1 + ui * fh2

            # Vertical flux
            fv1, fv2 = F(ui[1])
            un = ui * fv1 + field[n] * fv2
            us = field[s] * fv1 + ui * fv2

            # Total flux
            new_u = ui - kx * (ue - uw) - ky * (un - us)
            new_field[nid] = new_u

        return new_field
