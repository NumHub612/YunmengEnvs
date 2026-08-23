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
    OperatorMode,
)
from yunmeng.solvers.commons.solvers import BaseExplicitOperator, BaseImplicitOperator
from yunmeng.numerics.grids import Grid, ElementType
from yunmeng.numerics.algos import MeshTopo
from yunmeng.numerics.fields import (
    DataHub,
    DataProduct,
    Sample,
    Field,
    VariableType,
)
from yunmeng.solvers.commons.supports import backend_of_field


def _onesided_fill(out, u, kx, ky, backend):
    """Fill boundary slices of a gradient array with one-sided differences.

    Args:
        out: gradient array under construction, shape (nx, ny, ..., >=2) —
            the last axis indexes the derivative direction (0: d/dx, 1: d/dy)
            for scalar input, or the derivative direction slot for vector
            input handled by the caller.
        u: source array, shape (nx, ny, ...).
    Only the boundary slices are written; interior must be filled by caller.
    """
    # d/dx one-sided on i-edges
    out[0, ..., 0] = (u[1] - u[0]) * (2.0 * kx)
    out[-1, ..., 0] = (u[-1] - u[-2]) * (2.0 * kx)
    # d/dy one-sided on j-edges
    out[:, 0, ..., 1] = (u[:, 1] - u[:, 0]) * (2.0 * ky)
    out[:, -1, ..., 1] = (u[:, -1] - u[:, -2]) * (2.0 * ky)
    return out


class Grad01(BaseExplicitOperator):
    """
    First order upwind explicit gradient operator on structured grids.

    Technical features:
    + The upwind scheme prioritizes information from the fluid's upstream direction,
    introducing numerical dissipation that ensures strong computational stability.

    Use when:
    + It is primarily used to discretize convective terms in momentum equations,
    especially in high Reynolds number or convection-dominated flows,
    where it effectively prevents numerical oscillations.

    Backend-neutral and autograd-safe (stage-2 refactor).
    """

    @classmethod
    def get_type(cls) -> OperatorType:
        return OperatorType.GRAD

    @classmethod
    def get_name(cls) -> str:
        return "grad01"

    def __init__(self, target_fields: list[str]):
        super().__init__(target_fields)

        self._var = target_fields[0]
        self._dx = None
        self._dy = None

    def prepare(
        self,
        mesh: Grid,
        bounds: dict[str, list[IBoundaryCondition]],
    ):
        super().prepare(mesh, bounds)
        if not isinstance(mesh, Grid) or not mesh.uniform:
            raise ValueError(f"FDM op {self.get_name()} only supports uniform Grid.")

        self._dx = mesh.lx / (mesh.nx - 1)
        self._dy = mesh.ly / (mesh.ny - 1)

    def forward(self, data_hub: DataHub, time: float) -> Field:
        """Calculate the gradient of the field."""
        sample = data_hub.latest(self._var, ElementType.NODE)
        if sample is None:
            raise ValueError(f"Grad01: no data for '{self._var}'@NODE.")
        old_field = sample.data

        new_field = old_field.copy()
        self._apply_bc_directly(new_field, self._var)

        if old_field.vtype.is_scalar:
            grads = self._calculate_scalar_field(new_field)
        elif old_field.vtype.is_vector:
            grads = self._calculate_vector_field(new_field)
        else:
            raise ValueError(f"FDM op {self.get_name()} not support tensor fields.")

        self._publish(
            data_hub,
            self._var,
            ElementType.NODE,
            Sample(time, grads),
            self.get_type().value,
        )

        return grads

    def _calculate_scalar_field(self, field: Field) -> Field:
        """Vectorized upwind gradient for scalar field (backend-neutral)."""
        backend = backend_of_field(field)
        xp = backend.xp

        dim = self._mesh.dimension.value
        new_field = Field(field.mesh_shards, VariableType.vector(dim), field.etype)
        kx = 1.0 / self._dx
        ky = 1.0 / self._dy

        nx, ny = self._mesh.nx, self._mesh.ny
        u = field._shards[0].data.reshape(nx, ny)

        u_c = u[1:-1, 1:-1]
        u_e = u[2:, 1:-1]
        u_w = u[:-2, 1:-1]
        u_n = u[1:-1, 2:]
        u_s = u[1:-1, :-2]

        # Upwind coefficients (zero subgradient at the switching surface)
        fh1 = backend.where(u_c >= 0, 1.0, 0.0)
        fh2 = 1.0 - fh1
        fv1 = backend.where(u_c >= 0, 1.0, 0.0)
        fv2 = 1.0 - fv1

        ue = u_c * fh1 + u_e * fh2
        uw = u_w * fh1 + u_c * fh2
        un = u_c * fv1 + u_n * fv2
        us = u_s * fv1 + u_c * fv2

        ux = (ue - uw) * kx
        uy = (un - us) * ky

        grad = backend.zeros((nx, ny, dim), dtype=u.dtype)
        grad[1:-1, 1:-1, 0] = ux
        grad[1:-1, 1:-1, 1] = uy
        grad = _onesided_fill(grad, u, kx, ky, backend)

        new_field._shards[0].data = grad.reshape(-1, dim)
        return new_field

    def _calculate_vector_field(self, field: Field) -> Field:
        """Vectorized upwind gradient for vector field (backend-neutral)."""
        backend = backend_of_field(field)

        dim = field.vtype.shape[0]
        new_field = Field(field.mesh_shards, VariableType.tensor(dim), field.etype)
        kx = 1.0 / self._dx
        ky = 1.0 / self._dy

        nx, ny = self._mesh.nx, self._mesh.ny
        u = field._shards[0].data.reshape(nx, ny, dim)

        u_c = u[1:-1, 1:-1, :]  # (nx-2, ny-2, dim)
        u_e = u[2:, 1:-1, :]
        u_w = u[:-2, 1:-1, :]
        u_n = u[1:-1, 2:, :]
        u_s = u[1:-1, :-2, :]

        # Upwind coefficients based on flow direction per axis
        fh1 = backend.where(u_c[..., 0] >= 0, 1.0, 0.0)[..., None]
        fh2 = 1.0 - fh1
        fv1 = backend.where(u_c[..., 1] >= 0, 1.0, 0.0)[..., None]
        fv2 = 1.0 - fv1

        ue = u_c * fh1 + u_e * fh2
        uw = u_w * fh1 + u_c * fh2
        un = u_c * fv1 + u_n * fv2
        us = u_s * fv1 + u_c * fv2

        dudx = (ue[..., 0] - uw[..., 0]) * kx
        dudy = (un[..., 0] - us[..., 0]) * ky
        dvdx = (ue[..., 1] - uw[..., 1]) * kx
        dvdy = (un[..., 1] - us[..., 1]) * ky

        result = backend.zeros((nx, ny, dim, dim), dtype=u.dtype)
        result[1:-1, 1:-1, 0, 0] = dudx
        result[1:-1, 1:-1, 0, 1] = dudy
        result[1:-1, 1:-1, 1, 0] = dvdx
        result[1:-1, 1:-1, 1, 1] = dvdy

        # Vectorized one-sided boundary fill (graph-safe; filler values,
        # overwritten by solver-applied VALUE BCs after the update).
        # d/dx on i-edges, both components:
        result[0, :, 0, 0] = (u[1, :, 0] - u[0, :, 0]) * (2.0 * kx)
        result[0, :, 1, 0] = (u[1, :, 1] - u[0, :, 1]) * (2.0 * kx)
        result[-1, :, 0, 0] = (u[-1, :, 0] - u[-2, :, 0]) * (2.0 * kx)
        result[-1, :, 1, 0] = (u[-1, :, 1] - u[-2, :, 1]) * (2.0 * kx)
        # d/dy on j-edges:
        result[:, 0, 0, 1] = (u[:, 1, 0] - u[:, 0, 0]) * (2.0 * ky)
        result[:, 0, 1, 1] = (u[:, 1, 1] - u[:, 0, 1]) * (2.0 * ky)
        result[:, -1, 0, 1] = (u[:, -1, 0] - u[:, -2, 0]) * (2.0 * ky)
        result[:, -1, 1, 1] = (u[:, -1, 1] - u[:, -2, 1]) * (2.0 * ky)

        new_field._shards[0].data = result.reshape(-1, dim, dim)
        return new_field


class Grad02(BaseExplicitOperator):
    """
    Second order central explicit gradient operator on structured grids.

    Backend-neutral and autograd-safe (stage-2 refactor).
    """

    @classmethod
    def get_type(cls) -> OperatorType:
        return OperatorType.GRAD

    @classmethod
    def get_name(cls) -> str:
        return "grad02"

    def __init__(self, target_fields: list[str]):
        super().__init__(target_fields)
        self._var = target_fields[0]
        self._dx = None
        self._dy = None

    def prepare(
        self,
        mesh: Grid,
        bounds: dict[str, list[IBoundaryCondition]],
    ):
        super().prepare(mesh, bounds)
        if not isinstance(mesh, Grid) or not mesh.uniform:
            raise ValueError(f"FDM op {self.get_name()} only supports uniform Grid.")

        self._dx = mesh.lx / (mesh.nx - 1)
        self._dy = mesh.ly / (mesh.ny - 1)

    def forward(self, datahub: DataHub, time: float) -> Field:
        sample = datahub.latest(self._var, ElementType.NODE)
        if sample is None:
            raise ValueError(f"Grad02: no data for '{self._var}'@NODE.")
        old_field = sample.data
        new_field = old_field.copy()
        self._apply_bc_directly(new_field, self._var)

        if old_field.vtype.is_scalar:
            grads = self._calculate_scalar_field(new_field)
        elif old_field.vtype.is_vector:
            grads = self._calculate_vector_field(new_field)
        else:
            raise ValueError(f"FDM op {self.get_name()} not support tensor fields.")

        self._publish(
            datahub,
            self._var,
            ElementType.NODE,
            Sample(time, grads),
            self.get_type().value,
        )
        return grads

    def _calculate_vector_field(self, field: Field) -> Field:
        """Vectorized gradient for vector field (central difference)."""
        backend = backend_of_field(field)

        dim = field.vtype.shape[0]
        new_field = Field(field.mesh_shards, VariableType.tensor(dim), field.etype)
        kx = 1.0 / (2.0 * self._dx)
        ky = 1.0 / (2.0 * self._dy)

        nx, ny = self._mesh.nx, self._mesh.ny
        u = field._shards[0].data.reshape(nx, ny, dim)

        u_e = u[2:, 1:-1, :]
        u_w = u[:-2, 1:-1, :]
        u_n = u[1:-1, 2:, :]
        u_s = u[1:-1, :-2, :]

        dudx = (u_e[..., 0] - u_w[..., 0]) * kx
        dudy = (u_n[..., 0] - u_s[..., 0]) * ky
        dvdx = (u_e[..., 1] - u_w[..., 1]) * kx
        dvdy = (u_n[..., 1] - u_s[..., 1]) * ky

        result = backend.zeros((nx, ny, dim, dim), dtype=u.dtype)
        result[1:-1, 1:-1, 0, 0] = dudx
        result[1:-1, 1:-1, 0, 1] = dudy
        result[1:-1, 1:-1, 1, 0] = dvdx
        result[1:-1, 1:-1, 1, 1] = dvdy

        # Vectorized one-sided boundary fill (replaces the old per-node
        # Python loop, which breaks the autograd graph).
        result[0, :, 0, 0] = (u[1, :, 0] - u[0, :, 0]) / self._dx
        result[0, :, 1, 0] = (u[1, :, 1] - u[0, :, 1]) / self._dx
        result[-1, :, 0, 0] = (u[-1, :, 0] - u[-2, :, 0]) / self._dx
        result[-1, :, 1, 0] = (u[-1, :, 1] - u[-2, :, 1]) / self._dx
        result[:, 0, 0, 1] = (u[:, 1, 0] - u[:, 0, 0]) / self._dy
        result[:, 0, 1, 1] = (u[:, 1, 1] - u[:, 0, 1]) / self._dy
        result[:, -1, 0, 1] = (u[:, -1, 0] - u[:, -2, 0]) / self._dy
        result[:, -1, 1, 1] = (u[:, -1, 1] - u[:, -2, 1]) / self._dy

        new_field._shards[0].data = result.reshape(-1, dim, dim)
        return new_field

    def _calculate_scalar_field(self, field: Field) -> Field:
        """Vectorized gradient for scalar field (central difference)."""
        backend = backend_of_field(field)

        dim = self._mesh.dimension.value
        new_field = Field(field.mesh_shards, VariableType.vector(dim), field.etype)
        kx = 1.0 / (2.0 * self._dx)
        ky = 1.0 / (2.0 * self._dy)

        nx, ny = self._mesh.nx, self._mesh.ny
        u = field._shards[0].data.reshape(nx, ny)

        ux = (u[2:, 1:-1] - u[:-2, 1:-1]) * kx
        uy = (u[1:-1, 2:] - u[1:-1, :-2]) * ky

        grad = backend.zeros((nx, ny, dim), dtype=u.dtype)
        grad[1:-1, 1:-1, 0] = ux
        grad[1:-1, 1:-1, 1] = uy
        grad = _onesided_fill(grad, u, kx, ky, backend)

        new_field._shards[0].data = grad.reshape(-1, dim)
        return new_field
