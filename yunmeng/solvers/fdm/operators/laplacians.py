# -*- encoding: utf-8 -*-
"""
Copyright (C) 2025, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Laplacian operators for the finite difference method.

OPTIMIZED: Lap01 uses pre-allocated output + in-place modification
instead of creating new Field objects each call.
"""

from yunmeng.solvers.interfaces import (
    IBoundaryCondition,
    OperatorType,
    BoundaryType,
    OperatorMode,
)
from yunmeng.solvers.commons.solvers import BaseExplicitOperator, BaseImplicitOperator
from yunmeng.numerics.enums import BackendType
from yunmeng.numerics.grids import Grid, ElementType
from yunmeng.numerics.linalgs import LinearEqs, Matrix, NumpyMatrix, TorchMatrix
from yunmeng.numerics.fields import (
    DataHub,
    Sample,
    Field,
    Variable,
    VariableType,
)
import numpy as np


class Lap01(BaseExplicitOperator):
    """
    Center explicit scheme for laplacian operator.
    """

    @classmethod
    def get_type(cls) -> OperatorType:
        return OperatorType.LAPLACIAN

    @classmethod
    def get_name(cls) -> str:
        return "lap01"

    def __init__(self, fields: list[str], diffusivity: float = 1.0):
        super().__init__(fields)
        self._var = fields[0]
        self._nu = diffusivity
        self._dx = None
        self._dy = None

        # Pre-allocated output field
        self._out_field: Field = None

    def prepare(
        self,
        mesh: Grid,
        bounds: dict[str, list[IBoundaryCondition]],
    ):
        super().prepare(mesh, bounds)
        if not isinstance(mesh, Grid) or not mesh.uniform:
            raise ValueError(f"FDM op {self.get_name()} only supports uniform Grid.")

        self._dx = self._mesh.lx / (self._mesh.nx - 1)
        self._dy = self._mesh.ly / (self._mesh.ny - 1)

    def forward(self, datahub: DataHub, time: float) -> Field:
        sample = datahub.latest(self._var, ElementType.NODE)
        if sample is None:
            raise ValueError(f"Lap01: no data for '{self._var}'@NODE.")
        old_field = sample.data
        new_field = old_field.copy()
        self._apply_bc_directly(new_field, self._var)

        if old_field.vtype.is_scalar:
            result = self._calculate_scalar_field(new_field)
        elif old_field.vtype.is_vector:
            result = self._calculate_vector_field(new_field)
        else:
            raise ValueError("FDM op lap01 not support tensor fields.")

        # publish to cache
        self._publish(
            datahub,
            self._var,
            ElementType.NODE,
            Sample(time, result),
            self.get_type().value,
        )
        return result

    def _calculate_vector_field(self, field: Field) -> Field:
        """Vectorized vector field laplacian (5-point stencil)."""
        kx = self._nu / self._dx**2
        ky = self._nu / self._dy**2

        nx, ny = self._mesh.nx, self._mesh.ny
        dim = field._shards[0].data.shape[1]
        u = field._shards[0].data.reshape(nx, ny, dim)

        # Vectorized 5-point stencil on internal region
        lap = np.zeros_like(u)
        lap[1:-1, 1:-1, :] = (
            u[2:, 1:-1, :] - 2 * u[1:-1, 1:-1, :] + u[:-2, 1:-1, :]
        ) * kx + (u[1:-1, 2:, :] - 2 * u[1:-1, 1:-1, :] + u[1:-1, :-2, :]) * ky

        # Reuse pre-allocated field if possible, else create
        if self._out_field is None or self._out_field.vtype != VariableType.vector(dim):
            self._out_field = Field(
                field.mesh_shards, VariableType.vector(dim), field.etype
            )
        self._out_field._shards[0].data = lap.reshape(-1, dim)
        return self._out_field

    def _calculate_scalar_field(self, field: Field) -> Field:
        """Vectorized scalar field laplacian (5-point stencil)."""
        kx = self._nu / self._dx**2
        ky = self._nu / self._dy**2

        nx, ny = self._mesh.nx, self._mesh.ny
        u = field._shards[0].data.reshape(nx, ny)

        # Vectorized 5-point stencil on internal region
        lap = np.zeros_like(u)
        lap[1:-1, 1:-1] = (u[2:, 1:-1] - 2 * u[1:-1, 1:-1] + u[:-2, 1:-1]) * kx + (
            u[1:-1, 2:] - 2 * u[1:-1, 1:-1] + u[1:-1, :-2]
        ) * ky

        # Reuse pre-allocated field if possible
        if self._out_field is None or not self._out_field.vtype.is_scalar:
            self._out_field = Field(
                field.mesh_shards, VariableType.scalar(), field.etype
            )
        self._out_field._shards[0].data = lap.reshape(-1)
        return self._out_field


class Lap02(BaseImplicitOperator):
    """
    Center implicit scheme for laplacian operator on isotropic field.
    """

    @classmethod
    def get_type(cls) -> OperatorType:
        return OperatorType.LAPLACIAN

    @classmethod
    def get_name(cls) -> str:
        return "lap02"

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
        if not isinstance(mesh, Grid) or not mesh.uniform:
            raise ValueError(f"FDM op {self.get_name()} only supports uniform Grid.")

        self._dx = self._mesh.lx / (self._mesh.nx - 1)
        self._dy = self._mesh.ly / (self._mesh.ny - 1)

    def forward(self, datahub: DataHub, time: float = None) -> LinearEqs:
        sample = datahub.latest(self._var, ElementType.NODE)
        if sample is None:
            raise ValueError(f"Lap02: no data for '{self._var}'@NODE.")

        old_field = sample.data
        return self._generate_linear_eqs(old_field)

    def _create_matrix(self, field: Field) -> Matrix:
        """Create the matrix."""
        if field.meta.btype == BackendType.NUMPY:
            return NumpyMatrix
        elif field.meta.btype == BackendType.TORCH:
            return TorchMatrix
        else:
            raise ValueError("FDM op lap02 not support tensor fields.")

    def _generate_linear_eqs(self, field: Field) -> LinearEqs:
        """Calculate the vector field."""
        node_count = self._mesh.node_count
        values = np.zeros((node_count, node_count))
        rhs_arr = np.zeros(node_count)

        kx = 1.0 / self._dx**2
        ky = 1.0 / self._dy**2

        # Internal nodes: standard 5-point stencil
        for nid in self._topo.internal_nodes:
            e, w, n, s, _, _ = self._mesh.get_node_neighbours(nid)
            values[nid, nid] = -2 * (kx + ky)
            values[nid, e] = kx
            values[nid, w] = kx
            values[nid, n] = ky
            values[nid, s] = ky

        # Boundary nodes
        for nid in self._topo.boundary_nodes:
            bcs = self._bcs[self._var]
            e, w, n, s, _, _ = self._mesh.get_node_neighbours(nid)

            for bc in bcs:
                if not bc.region.include(nid, ElementType.NODE):
                    continue

                if bc.get_type() == BoundaryType.VALUE:
                    # Dirichlet: enforce p = value at boundary
                    values[nid, :] = 0.0
                    values[nid, nid] = 1.0
                    val = bc.get().value
                    if isinstance(val, Variable):
                        val = val.data
                    if hasattr(val, "item"):
                        val = val.item()
                    rhs_arr[nid] = float(val)
                elif bc.get_type() == BoundaryType.FLUX:
                    # Neumann: ghost node reflection (2nd-order)
                    flux = bc.get().flux
                    qx, qy = self._extract_flux_components(flux)

                    # Clear the row
                    values[nid, :] = 0.0

                    diag_coeff = 0.0

                    # Horizontal direction
                    if w is None and e is not None:
                        diag_coeff += -2 * kx
                        values[nid, e] = 2 * kx
                        rhs_arr[nid] += -2 * qx / self._dx
                    elif e is None and w is not None:
                        diag_coeff += -2 * kx
                        values[nid, w] = 2 * kx
                        rhs_arr[nid] += 2 * qx / self._dx
                    elif e is not None and w is not None:
                        diag_coeff += -2 * kx
                        values[nid, e] = kx
                        values[nid, w] = kx

                    # Vertical direction
                    if s is None and n is not None:
                        diag_coeff += -2 * ky
                        values[nid, n] = 2 * ky
                        rhs_arr[nid] += -2 * qy / self._dy
                    elif n is None and s is not None:
                        diag_coeff += -2 * ky
                        values[nid, s] = 2 * ky
                        rhs_arr[nid] += 2 * qy / self._dy
                    elif n is not None and s is not None:
                        diag_coeff += -2 * ky
                        values[nid, n] = ky
                        values[nid, s] = ky

                    values[nid, nid] = diag_coeff

        # Assemble linear system
        matrix = self._create_matrix(field).from_data(values)

        # Build RHS field from array
        rhs_field = Field.from_array(
            rhs_arr,
            field.mesh_shards,
            VariableType.scalar(),
            field.etype,
        )

        eqs = LinearEqs(matrix, rhs_field)
        return eqs

    @staticmethod
    def _extract_flux_components(flux) -> tuple[float, float]:
        """Extract (qx, qy) from a flux value (Variable, list, or scalar)."""
        qx, qy = 0.0, 0.0
        if isinstance(flux, Variable):
            data = flux.data
            if flux.vtype.is_scalar:
                qx = qy = float(data)
            elif flux.vtype.is_vector:
                qx = float(data[0])
                qy = float(data[1])
        elif isinstance(flux, (list, tuple, np.ndarray)):
            arr = np.asarray(flux).flatten()
            qx = float(arr[0]) if len(arr) >= 1 else 0.0
            qy = float(arr[1]) if len(arr) >= 2 else 0.0
        elif np.isscalar(flux):
            qx = qy = float(flux)
        return qx, qy
