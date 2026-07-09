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
    OperatorMode,
)
from yunmeng.solvers.commons.solvers import BaseExplicitOperator, BaseImplicitOperator
from yunmeng.numerics.grids import Grid, ElementType
from yunmeng.numerics.fields import (
    DataHub,
    DataProduct,
    Sample,
    Field,
    Variable,
    VariableType,
)

import numpy as np


class Div01(BaseExplicitOperator):
    """
    Divergence operator based on central difference.
    """

    @classmethod
    def get_type(cls) -> OperatorType:
        return OperatorType.DIV

    @classmethod
    def get_name(cls) -> str:
        return "div01"

    @classmethod
    def consumes(cls, field_name: str, etype: ElementType) -> list[DataProduct]:
        """Declare: div operater can reuse a pre-computed gradient."""
        return [
            DataProduct(
                OperatorType.GRAD.value,
                field_name,
                etype,
                namespace="*",
            )
        ]

    def __init__(self, target_fields: list[str]):
        super().__init__(target_fields)
        self._var = target_fields[0]
        self._dx = None
        self._dy = None

    def prepare(
        self,
        mesh: Grid,
        bounds: dict[str, list[IBoundaryCondition]] = None,
    ):
        super().prepare(mesh, bounds)
        if not isinstance(mesh, Grid):
            raise ValueError("FDM op div01 only supports Grid.")

        self._dx = self._mesh.lx / (self._mesh.nx - 1)
        self._dy = self._mesh.ly / (self._mesh.ny - 1)

    def forward(self, datahub: DataHub, time: float) -> Field:
        """Calculate the divergence of the field.

        If a grad operator has already published grad(u) to the cache,
        div compute divergence as trace(grad_u) instead of
        re-doing finite differences.
        """
        # 1) Try to reuse cached gradient
        cached_grad = self._query(datahub, self._var, ElementType.NODE, "grad")
        if cached_grad is not None:
            return self._div_from_gradient(cached_grad.data)

        # 2) Fallback: extract field and compute from scratch
        sample = datahub.latest(self._var, ElementType.NODE)
        if sample is None:
            raise ValueError(f"Div01: no data for '{self._var}'@NODE.")
        old_field = sample.data
        div = self._calculate_divergence(old_field)

        # Publish to cache
        self._publish(
            datahub,
            self._var,
            ElementType.NODE,
            Sample(time, div),
            self.get_type().value,
        )
        return div

    def _div_from_gradient(self, grad_field: Field) -> Field:
        """Compute divergence as trace of the gradient tensor: div(u) = dudx + dvdy.

        grad_field layout (for 2D vector): (N, 2, 2) where
          grad[..., 0, 0] = dudx,  grad[..., 0, 1] = dudy
          grad[..., 1, 0] = dvdx,  grad[..., 1, 1] = dvdy
        """
        div = Field(grad_field.mesh_shards, VariableType.scalar(), grad_field.etype)
        g = grad_field._shards[0].data  # (N, 2, 2) or (N,)

        if g.ndim == 3 and g.shape[1:] == (2, 2):
            # Vector field gradient: trace = dudx + dvdy
            div._shards[0].data = g[:, 0, 0] + g[:, 1, 1]
        elif g.ndim == 2 and g.shape[1] == 2:
            # Scalar field gradient: divergence not defined directly,
            # but for consistency treat as the sum (rare case)
            div._shards[0].data = g[:, 0] + g[:, 1]
        else:
            raise ValueError(
                f"Div01: unexpected gradient shape {g.shape}, cannot compute trace"
            )
        return div

    def _calculate_divergence(self, u_field: Field) -> Field:
        """Vectorized divergence div(u) = du/dx + dv/dy."""
        div = Field(
            u_field.mesh_shards,
            VariableType.scalar(),
            u_field.etype,
        )

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
        for bc in self._bcs[self._var]:
            if bc.get_type() == BoundaryType.VALUE:
                bc.apply(u_bc)

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
