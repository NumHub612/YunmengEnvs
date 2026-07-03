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
from yunmeng.numerics.grids import Grid, ElementType
from yunmeng.numerics.algos import MeshTopo
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
        self._mesh: Grid = None
        self._topo: MeshTopo = None
        self._bcs = None

        self._dx = None
        self._dy = None

    @property
    def target_fields(self) -> list[str]:
        return [self._var]

    def prepare(
        self,
        mesh: Grid,
        bounds: dict[int, dict[str, IBoundaryCondition]] = None,
    ):
        if not isinstance(mesh, Grid):
            raise ValueError("FDM op grad02 only supports Grid.")
        self._mesh = mesh
        self._bcs = bounds

        self._topo = self._mesh.get_topo_assistant()
        self._dx = self._mesh.lx / (self._mesh.nx - 1)
        self._dy = self._mesh.ly / (self._mesh.ny - 1)

    def forward(self, sources: Field | DataHub, dt: float = None) -> Field:
        """Calculate the divergence of the field."""
        if isinstance(sources, Field):
            old_field = sources
        else:
            old_field = sources.field(self._var, loc=ElementType.NODE).data
        if len(old_field.mesh_shards) != 1:
            # TODO: Support multi-gpu gradient operator
            raise ValueError(f"FDM op {self.get_name()} only supports cpu.")

        return self._calculate_divergence(old_field)

    def _calculate_divergence(self, u_field: Field) -> Field:
        """直接基于速度场中心差分计算 div(u) = du/dx + dv/dy"""
        div = Field(u_field.mesh_shards, VariableType.scalar(), u_field.etype)

        u_bc = u_field.copy()
        for nid in self._topo.boundary_nodes:
            bc = self._bcs[nid]["u"]
            if bc.get_type() == BoundaryType.VALUE:
                u_bc[nid] = bc.evaluate().value

        kx = 1.0 / (2.0 * self._dx)
        ky = 1.0 / (2.0 * self._dy)

        for nid in range(self._mesh.node_count):
            e, w, n, s, _, _ = self._mesh.get_node_neighbours(nid)
            ue = u_bc[e] if e is not None else u_bc[nid]
            uw = u_bc[w] if w is not None else u_bc[nid]
            un = u_bc[n] if n is not None else u_bc[nid]
            us = u_bc[s] if s is not None else u_bc[nid]

            dudx = (ue[0] - uw[0]) * kx
            dvdy = (un[1] - us[1]) * ky
            div[nid] = Variable.scalar(dudx + dvdy)

        return div
