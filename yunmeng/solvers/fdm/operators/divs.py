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

import numpy as np


class Div01(IOperator):
    """
    Divergence operator based on central difference.
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
            raise ValueError("FDM op div01 only supports Grid.")
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
            raise ValueError(f"FDM op {self.get_name()} only supports cpu.")

        return self._calculate_divergence(old_field)

    def _calculate_divergence(self, u_field: Field) -> Field:
        """Vectorized divergence div(u) = du/dx + dv/dy."""
        div = Field(u_field.mesh_shards, VariableType.scalar(), u_field.etype)

        nx, ny = self._mesh.nx, self._mesh.ny
        # (N, 2) -> (nx, ny, 2)
        u = u_field._shards[0].data.reshape(nx, ny, 2)

        kx = 1.0 / (2.0 * self._dx)
        ky = 1.0 / (2.0 * self._dy)

        # --- Internal region: vectorized central difference ---
        u_e = u[2:, 1:-1, :]
        u_w = u[:-2, 1:-1, :]
        u_n = u[1:-1, 2:, :]
        u_s = u[1:-1, :-2, :]

        # div(u) = du/dx + dv/dy
        dudx = (u_e[..., 0] - u_w[..., 0]) * kx
        dvdy = (u_n[..., 1] - u_s[..., 1]) * ky

        result = np.zeros((nx, ny))
        result[1:-1, 1:-1] = dudx + dvdy

        # Write internal region
        div._shards[0].data = result.reshape(-1)

        # --- Boundary nodes: apply BC then compute ---
        u_bc = u_field.copy()
        for nid in self._topo.boundary_nodes:
            bc = self._bcs[nid]["u"]
            if bc.get_type() == BoundaryType.VALUE:
                u_bc[nid] = bc.evaluate().value

        for nid in self._topo.boundary_nodes:
            e, w, n, s, _, _ = self._mesh.get_node_neighbours(nid)
            ue = u_bc[e] if e is not None else u_bc[nid]
            uw = u_bc[w] if w is not None else u_bc[nid]
            un = u_bc[n] if n is not None else u_bc[nid]
            us = u_bc[s] if s is not None else u_bc[nid]

            dudx_b = (ue[0] - uw[0]) * kx
            dvdy_b = (un[1] - us[1]) * ky
            div[nid] = Variable.scalar(dudx_b + dvdy_b)

        return div
