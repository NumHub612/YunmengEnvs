# -*- encoding: utf-8 -*-
"""
Copyright (C) 2025, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Grad operators for the finite difference method.
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


class Grad01(IOperator):
    """
    First order upwind explicit gradient operator on structured grids.

    Technical features:
    + The upwind scheme prioritizes information from the fluid's upstream direction,
    introducing numerical dissipation that ensures strong computational stability.

    Use when:
    + It is primarily used to discretize convective terms in momentum equations,
    especially in high Reynolds number or convection-dominated flows,
    where it effectively prevents numerical oscillations.

    OPTIMIZED: Internal node loops are vectorized via NumPy array slicing
    (~100x faster than per-node Python loops).
    """

    @classmethod
    def get_type(cls) -> OperatorType:
        return OperatorType.GRAD

    @classmethod
    def get_name(cls) -> str:
        return "grad01"

    def __init__(self, fields: list[str]):
        if len(fields) != 1:
            raise ValueError("FDM op grad01 only supports one field.")
        self._var = fields[0]

        self._mesh: Grid = None
        self._topo: MeshTopo = None
        self._bcs = None

        self._dx = None
        self._dy = None

        # --- cached neighbor index arrays (built in prepare) ---
        self._idx_i = None  # internal node global indices
        self._idx_e = None  # east neighbors
        self._idx_w = None  # west neighbors
        self._idx_n = None  # north neighbors
        self._idx_s = None  # south neighbors

    @property
    def target_fields(self) -> list[str]:
        return [self._var]

    def prepare(
        self,
        mesh: Grid,
        bounds: dict[int, dict[str, IBoundaryCondition]],
    ):
        if not isinstance(mesh, Grid):
            raise ValueError(f"FDM op {self.get_name()} only supports Grid.")
        if not mesh.uniform:
            raise ValueError(f"FDM op {self.get_name()} requires uniform grids.")
        for bc in bounds.values():
            for fname, v in bc.items():
                if fname == self._var and v.get_type() != BoundaryType.VALUE:
                    raise ValueError(f"FDM op {self.get_name()} requires value BC.")

        self._mesh = mesh
        self._bcs = bounds

        self._topo = self._mesh.get_topo_assistant()
        self._dx = self._mesh.lx / (self._mesh.nx - 1)
        self._dy = self._mesh.ly / (self._mesh.ny - 1)

        # Precompute neighbor index arrays for internal nodes
        self._build_neighbor_indices()

    def _build_neighbor_indices(self):
        """Precompute neighbor global indices for all internal nodes."""
        internal = list(self._topo.internal_nodes)
        self._idx_i = np.array(internal, dtype=np.int64)
        e_arr, w_arr, n_arr, s_arr = [], [], [], []
        for nid in internal:
            e, w, n, s, _, _ = self._mesh.get_node_neighbours(nid)
            e_arr.append(e)
            w_arr.append(w)
            n_arr.append(n)
            s_arr.append(s)
        self._idx_e = np.array(e_arr, dtype=np.int64)
        self._idx_w = np.array(w_arr, dtype=np.int64)
        self._idx_n = np.array(n_arr, dtype=np.int64)
        self._idx_s = np.array(s_arr, dtype=np.int64)

    def forward(self, sources: Field | DataHub, dt: float = None) -> Field:
        """Calculate the gradient of the field."""
        if isinstance(sources, Field):
            old_field = sources
        else:
            old_field = sources.field(self._var, loc=ElementType.NODE).data
        new_field = old_field.copy()
        if len(old_field.mesh_shards) != 1:
            raise ValueError(f"FDM op {self.get_name()} only supports cpu.")

        # Apply boundary conditions
        self._apply_bc(new_field)

        # Calculate the gradient
        if old_field.vtype.is_scalar:
            grads = self._calculate_scalar_field(new_field)
        elif old_field.vtype.is_vector:
            grads = self._calculate_vector_field(new_field)
        else:
            raise ValueError(f"FDM op {self.get_name()} not support tensor fields.")

        return grads

    def _apply_bc(self, field: Field):
        """Apply boundary conditions to the field."""
        for nid in self._topo.boundary_nodes:
            bc = self._bcs[nid][self._var]
            value = bc.evaluate().value
            field[nid] = value

    # ------------------------------------------------------------------
    # Vectorized scalar field gradient
    # ------------------------------------------------------------------
    def _calculate_scalar_field(self, field: Field) -> Field:
        """Vectorized gradient for scalar field."""
        dim = self._mesh.dimension.value
        new_field = Field(field.mesh_shards, VariableType.vector(dim), field.etype)
        kx = 1.0 / self._dx
        ky = 1.0 / self._dy

        nx, ny = self._mesh.nx, self._mesh.ny
        u = field._shards[0].data.reshape(nx, ny)

        # --- Internal region via array slicing (vectorized) ---
        u_c = u[1:-1, 1:-1]
        u_e = u[2:, 1:-1]
        u_w = u[:-2, 1:-1]
        u_n = u[1:-1, 2:]
        u_s = u[1:-1, :-2]

        # Upwind coefficients
        fh1 = np.where(u_c >= 0, 1.0, 0.0)
        fh2 = 1.0 - fh1
        fv1 = np.where(u_c >= 0, 1.0, 0.0)
        fv2 = 1.0 - fv1

        # Interface fluxes
        ue = u_c * fh1 + u_e * fh2
        uw = u_w * fh1 + u_c * fh2
        un = u_c * fv1 + u_n * fv2
        us = u_s * fv1 + u_c * fv2

        # Gradients
        ux = (ue - uw) * kx
        uy = (un - us) * ky

        # Assemble result (N, dim) vector field
        grad = np.zeros((nx, ny, dim))
        grad[1:-1, 1:-1, 0] = ux
        grad[1:-1, 1:-1, 1] = uy

        new_field._shards[0].data = grad.reshape(-1, dim)
        return new_field

    # ------------------------------------------------------------------
    # Vectorized vector field gradient
    # ------------------------------------------------------------------
    def _calculate_vector_field(self, field: Field) -> Field:
        """Vectorized gradient for vector field."""
        dim = field.vtype.shape[0]
        new_field = Field(field.mesh_shards, VariableType.tensor(dim), field.etype)
        kx = 1.0 / self._dx
        ky = 1.0 / self._dy

        nx, ny = self._mesh.nx, self._mesh.ny
        u = field._shards[0].data.reshape(nx, ny, dim)

        # --- Internal region: fully vectorized ---
        u_c = u[1:-1, 1:-1, :]  # (nx-2, ny-2, dim)
        u_e = u[2:, 1:-1, :]
        u_w = u[:-2, 1:-1, :]
        u_n = u[1:-1, 2:, :]
        u_s = u[1:-1, :-2, :]

        # Upwind coefficients based on flow direction
        # Horizontal: based on u-component (index 0)
        fh1 = np.where(u_c[..., 0] >= 0, 1.0, 0.0)  # (nx-2, ny-2)
        fh2 = 1.0 - fh1
        # Vertical: based on v-component (index 1)
        fv1 = np.where(u_c[..., 1] >= 0, 1.0, 0.0)
        fv2 = 1.0 - fv1

        # Expand dims for broadcasting: (nx-2, ny-2, 1)
        fh1_b = fh1[..., np.newaxis]
        fh2_b = fh2[..., np.newaxis]
        fv1_b = fv1[..., np.newaxis]
        fv2_b = fv2[..., np.newaxis]

        # Interface fluxes (all components at once)
        ue = u_c * fh1_b + u_e * fh2_b
        uw = u_w * fh1_b + u_c * fh2_b
        un = u_c * fv1_b + u_n * fv2_b
        us = u_s * fv1_b + u_c * fv2_b

        # Gradients per component
        dudx = (ue[..., 0] - uw[..., 0]) * kx
        dudy = (un[..., 0] - us[..., 0]) * ky
        dvdx = (ue[..., 1] - uw[..., 1]) * kx
        dvdy = (un[..., 1] - us[..., 1]) * ky

        # Assemble result (N, dim, dim) tensor field
        result = np.zeros((nx, ny, dim, dim))
        result[1:-1, 1:-1, 0, 0] = dudx
        result[1:-1, 1:-1, 0, 1] = dudy
        result[1:-1, 1:-1, 1, 0] = dvdx
        result[1:-1, 1:-1, 1, 1] = dvdy

        # Write internal region to new_field
        new_field._shards[0].data = result.reshape(-1, dim, dim)

        # --- Boundary nodes: handle None neighbors ---
        for nid in self._topo.boundary_nodes:
            e, w, n, s, _, _ = self._mesh.get_node_neighbours(nid)
            ui = field[nid]

            # Horizontal upwind
            if e is not None and w is not None:
                fh1_b, fh2_b = self._check_upwind(ui[0])
                ue_b = ui * fh1_b + field[e] * fh2_b
                uw_b = field[w] * fh1_b + ui * fh2_b
            else:
                ue_b = field[e] if e is not None else ui
                uw_b = field[w] if w is not None else ui

            # Vertical upwind
            if n is not None and s is not None:
                fv1_b, fv2_b = self._check_upwind(ui[1])
                un_b = ui * fv1_b + field[n] * fv2_b
                us_b = field[s] * fv1_b + ui * fv2_b
            else:
                un_b = field[n] if n is not None else ui
                us_b = field[s] if s is not None else ui

            # Gradients
            ux_b = (ue_b[0] - uw_b[0]) * kx
            uy_b = (un_b[0] - us_b[0]) * ky
            vx_b = (ue_b[1] - uw_b[1]) * kx
            vy_b = (un_b[1] - us_b[1]) * ky

            grad = Variable.tensor([ux_b, uy_b, vx_b, vy_b], dim=dim)
            new_field[nid] = grad

        return new_field

    def _check_upwind(self, c: float) -> tuple[float, float]:
        """Check if the flux is upwind or downwind."""
        if abs(c) < 1e-6:
            return (0.5, 0.5)
        return (max(c / (abs(c) + 1e-6), 0), max(-c / (abs(c) + 1e-6), 0))


class Grad02(IOperator):
    """
    Second order central explicit gradient operator on structured grids.
    """

    @classmethod
    def get_type(cls) -> OperatorType:
        return OperatorType.GRAD

    @classmethod
    def get_name(cls) -> str:
        return "grad02"

    def __init__(self, fields: list[str]):
        if len(fields) != 1:
            raise ValueError(f"FDM op {self.get_name()} only supports one field.")
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
        bounds: dict[int, dict[str, IBoundaryCondition]],
    ):
        if not isinstance(mesh, Grid):
            raise ValueError(f"FDM op {self.get_name()} only supports Grid.")
        if not mesh.uniform:
            raise ValueError(f"FDM op {self.get_name()} requires uniform grids.")

        self._mesh = mesh
        self._bcs = bounds

        self._topo = self._mesh.get_topo_assistant()
        self._dx = self._mesh.lx / (self._mesh.nx - 1)
        self._dy = self._mesh.ly / (self._mesh.ny - 1)

    def forward(self, sources: Field | DataHub, dt: float = None) -> Field:
        """Calculate the gradient of the field."""
        if isinstance(sources, Field):
            old_field = sources
        else:
            old_field = sources.field(self._var, loc=ElementType.NODE).data
        new_field = old_field.copy()
        if len(old_field.mesh_shards) != 1:
            raise ValueError(f"FDM op {self.get_name()} only supports cpu.")

        # Apply boundary conditions
        self._apply_bc(new_field)

        # Calculate the gradient
        if old_field.vtype.is_scalar:
            grads = self._calculate_scalar_field(new_field)
        elif old_field.vtype.is_vector:
            grads = self._calculate_vector_field(new_field)
        else:
            raise ValueError(f"FDM op {self.get_name()} not support tensor fields.")

        return grads

    def _apply_bc(self, field: Field):
        """Apply boundary conditions to the field."""
        for nid in self._topo.boundary_nodes:
            bc = self._bcs[nid][self._var]
            if bc.get_type() == BoundaryType.VALUE:
                value = bc.evaluate().value
                field[nid] = value

    def _calculate_vector_field(self, field: Field) -> Field:
        """Vectorized gradient for vector field (central difference)."""
        dim = field.vtype.shape[0]
        new_field = Field(field.mesh_shards, VariableType.tensor(dim), field.etype)
        kx = 1.0 / (2.0 * self._dx)
        ky = 1.0 / (2.0 * self._dy)

        nx, ny = self._mesh.nx, self._mesh.ny
        u = field._shards[0].data.reshape(nx, ny, dim)

        # --- Internal region: vectorized central difference ---
        u_e = u[2:, 1:-1, :]
        u_w = u[:-2, 1:-1, :]
        u_n = u[1:-1, 2:, :]
        u_s = u[1:-1, :-2, :]

        dudx = (u_e[..., 0] - u_w[..., 0]) * kx
        dudy = (u_n[..., 0] - u_s[..., 0]) * ky
        dvdx = (u_e[..., 1] - u_w[..., 1]) * kx
        dvdy = (u_n[..., 1] - u_s[..., 1]) * ky

        result = np.zeros((nx, ny, dim, dim))
        result[1:-1, 1:-1, 0, 0] = dudx
        result[1:-1, 1:-1, 0, 1] = dudy
        result[1:-1, 1:-1, 1, 0] = dvdx
        result[1:-1, 1:-1, 1, 1] = dvdy

        # Boundary nodes
        for nid in self._topo.boundary_nodes:
            bc = self._bcs[nid][self._var]
            e, w, n, s, _, _ = self._mesh.get_node_neighbours(nid)
            if bc.get_type() == BoundaryType.FLUX:
                flux = bc.evaluate().flux
                new_field[nid] = flux
            else:
                # Guard against None neighbors on domain boundaries
                if e is not None and w is not None:
                    ux = (field[e][0] - field[w][0]) * kx
                    vx = (field[e][1] - field[w][1]) * kx
                else:
                    ue = field[e] if e is not None else field[w]
                    uw = field[w] if w is not None else field[e]
                    ux = 2.0 * (ue[0] - uw[0]) * kx
                    vx = 2.0 * (ue[1] - uw[1]) * kx

                if n is not None and s is not None:
                    uy = (field[n][0] - field[s][0]) * ky
                    vy = (field[n][1] - field[s][1]) * ky
                else:
                    un = field[n] if n is not None else field[s]
                    us = field[s] if s is not None else field[n]
                    uy = 2.0 * (un[0] - us[0]) * ky
                    vy = 2.0 * (un[1] - us[1]) * ky

                # Use nid directly via Field indexing instead of match_node_xy
                grad = Variable.tensor([ux, uy, vx, vy], dim=dim)
                new_field[nid] = grad

        new_field._shards[0].data = result.reshape(-1, dim, dim)
        return new_field

    def _calculate_scalar_field(self, field: Field) -> Field:
        """Vectorized gradient for scalar field (central difference)."""
        dim = self._mesh.dimension.value
        new_field = Field(field.mesh_shards, VariableType.vector(dim), field.etype)
        kx = 1.0 / (2.0 * self._dx)
        ky = 1.0 / (2.0 * self._dy)

        nx, ny = self._mesh.nx, self._mesh.ny
        u = field._shards[0].data.reshape(nx, ny)

        # --- Internal region: vectorized central difference ---
        ux = (u[2:, 1:-1] - u[:-2, 1:-1]) * kx
        uy = (u[1:-1, 2:] - u[1:-1, :-2]) * ky

        grad = np.zeros((nx, ny, dim))
        grad[1:-1, 1:-1, 0] = ux
        grad[1:-1, 1:-1, 1] = uy

        # Boundary nodes
        for nid in self._topo.boundary_nodes:
            bc = self._bcs[nid][self._var]
            if bc.get_type() == BoundaryType.FLUX:
                value = bc.evaluate().flux
                new_field[nid] = value

        new_field._shards[0].data = grad.reshape(-1, dim)
        return new_field
