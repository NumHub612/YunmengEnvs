# -*- encoding: utf-8 -*-
"""
Copyright (C) 2025, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Laplacian operators for the finite difference method.
"""

from yunmeng.solvers.interfaces import (
    IBoundaryCondition,
    IOperator,
    OperatorType,
    BoundaryType,
)
from yunmeng.numerics.enums import BackendType
from yunmeng.numerics.grids import Grid, ElementType
from yunmeng.numerics.algos import MeshTopo
from yunmeng.numerics.mats import LinearEqs, Matrix, NumpyMatrix, TorchMatrix
from yunmeng.numerics.fields import DataHub, Field, VariableType, Variable
import numpy as np


class Lap01(IOperator):
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
        if len(fields) != 1:
            raise ValueError("FDM op lap01 only supports one field.")
        self._var = fields[0]

        self._mesh: Grid = None
        self._topo: MeshTopo = None
        self._bcs = None

        self._nu = diffusivity
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
            raise ValueError("FDM op lap01 only supports Grid.")
        if not mesh.uniform:
            # TODO: Support non-uniform grids
            raise ValueError("FDM op lap01 requires uniform grids.")

        for bc in bounds.values():
            # TODO: Support more types of boundary conditions
            for fname, v in bc.items():
                if fname == self._var and v.get_type() != BoundaryType.VALUE:
                    raise ValueError("FDM op lap01 requires value BC.")

        self._mesh = mesh
        self._bcs = bounds

        self._topo = self._mesh.get_topo_assistant()
        self._dx = self._mesh.lx / (self._mesh.nx - 1)
        self._dy = self._mesh.ly / (self._mesh.ny - 1)

    def run(self, sources: Field | DataHub, dt: float = None) -> Field:
        if isinstance(sources, Field):
            old_field = sources
        else:
            old_field = sources.field(self._var, loc=ElementType.NODE).data
        new_field = old_field.copy()
        if len(old_field.mesh_shards) != 1:
            # TODO: Support multi-gpu divergence operator
            raise ValueError("FDM op lap01 only supports cpu.")

        # Apply boundary conditions
        self._apply_bc(new_field)

        # Run laplacian operator
        if old_field.vtype == VariableType.SCALAR:
            new_field = self._calculate_scalar_field(new_field)
        elif old_field.vtype == VariableType.VECTOR:
            new_field = self._calculate_vector_field(new_field)
        else:
            raise ValueError("FDM op lap01 not support tensor fields.")

        return new_field

    def _apply_bc(self, field: Field):
        """Apply boundary conditions to the field."""
        for nid in self._topo.boundary_nodes:
            bc = self._bcs[nid][self._var]
            value = bc.evaluate().value
            field[nid] = value

    def _calculate_vector_field(self, field: Field) -> Field:
        """Calculate the vector field."""
        new_field = Field(field.mesh_shards, VariableType.VECTOR, field.etype)
        kx = self._nu / self._dx**2
        ky = self._nu / self._dy**2

        for nid in self._topo.internal_nodes:
            # Neighbour nodes
            e, w, n, s, _, _ = self._mesh.get_node_neighbours(nid)
            ue, uw, un, us, ui = field[[e, w, n, s, nid]]

            # Horizontal result
            uh = (ue - 2 * ui + uw) * kx

            # Vertical result
            uv = (un - 2 * ui + us) * ky

            new_field[nid] = uh + uv

        for nid in self._topo.boundary_nodes:
            # Neighbour nodes
            e, w, n, s, _, _ = self._mesh.get_node_neighbours(nid)
            ui = field[nid]

            # Horizontal result
            ue = field[e] if e is not None else ui
            uw = field[w] if w is not None else ui
            uh = (ue - 2 * ui + uw) * kx

            # Vertical result
            un = field[n] if n is not None else ui
            us = field[s] if s is not None else ui
            uv = (un - 2 * ui + us) * ky

            new_field[nid] = uh + uv

        return new_field

    def _calculate_scalar_field(self, field: Field) -> Field:
        """Calculate the scalar field."""
        new_field = Field(field.mesh_shards, VariableType.SCALAR, field.etype)
        kx = self._nu / self._dx**2
        ky = self._nu / self._dy**2

        for nid in self._topo.internal_nodes:
            # Neighbour nodes
            e, w, n, s, _, _ = self._mesh.get_node_neighbours(nid)
            ue, uw, un, us, ui = field[[e, w, n, s, nid]]

            # Horizontal result
            uh = (ue - 2 * ui + uw) * kx

            # Vertical result
            uv = (un - 2 * ui + us) * ky

            new_field[nid] = uh + uv

        return new_field


class Lap02(IOperator):
    """
    Center implicit scheme for laplacian operator on isotropic field.
    """

    @classmethod
    def get_type(cls) -> OperatorType:
        return OperatorType.LAPLACIAN

    @classmethod
    def get_name(cls) -> str:
        return "lap02"

    def __init__(self, fields: list[str]):
        if len(fields) != 1:
            raise ValueError("FDM op lap02 only supports one field.")
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
        self, mesh: Grid, bounds: dict[int, dict[str, IBoundaryCondition]] = None
    ):
        if not isinstance(mesh, Grid):
            raise ValueError("FDM op lap02 only supports Grid.")
        if not mesh.uniform:
            # TODO: Support non-uniform grids
            raise ValueError("FDM op lap02 requires uniform grids.")

        self._bcs = bounds
        self._mesh = mesh
        self._topo = self._mesh.get_topo_assistant()
        self._dx = self._mesh.lx / (self._mesh.nx - 1)
        self._dy = self._mesh.ly / (self._mesh.ny - 1)

    def run(self, sources: Field | DataHub, dt: float = None) -> LinearEqs:
        if isinstance(sources, Field):
            old_field = sources
        else:
            old_field = sources.field(self._var, loc=ElementType.NODE).data
        if len(old_field.mesh_shards) != 1:
            # TODO: Support multi-gpu divergence operator
            raise ValueError("FDM op lap02 only supports cpu.")

        eqs = self._generate_linear_eqs(old_field)
        return eqs

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

        # --------------------------------------------------
        # Internal nodes: standard 5-point stencil
        # --------------------------------------------------
        for nid in self._topo.internal_nodes:
            e, w, n, s, _, _ = self._mesh.get_node_neighbours(nid)
            values[nid, nid] = -2 * (kx + ky)
            values[nid, e] = kx
            values[nid, w] = kx
            values[nid, n] = ky
            values[nid, s] = ky

        # --------------------------------------------------
        # Boundary nodes
        # --------------------------------------------------
        for nid in self._topo.boundary_nodes:
            bc = self._bcs[nid][self._var]
            e, w, n, s, _, _ = self._mesh.get_node_neighbours(nid)

            if bc.get_type() == BoundaryType.VALUE:
                # --------------------------------------------------
                # Dirichlet: enforce p = value at boundary
                # --------------------------------------------------
                values[nid, :] = 0.0
                values[nid, nid] = 1.0
                val = bc.evaluate().value
                if isinstance(val, Variable):
                    val = val.data if val.vtype == VariableType.SCALAR else val.data
                rhs_arr[nid] = float(val)

            elif bc.get_type() == BoundaryType.FLUX:
                # --------------------------------------------------
                # Neumann: ghost node reflection (2nd-order)
                #
                # For ∂p/∂n = q at a boundary:
                #   - Set ghost node via: p_ghost = p_inner ∓ 2*dx*q
                #   - This doubles the inner neighbor coefficient
                #   - Adds a source term ±2*q/dx to RHS
                #
                # When q = 0 (zero flux), this reduces to symmetric reflection:
                #   p_ghost = p_inner, giving 2nd-order accurate Laplacian.
                # --------------------------------------------------
                flux = bc.evaluate().flux
                qx, qy = self._extract_flux_components(flux)

                # Clear the row
                values[nid, :] = 0.0

                diag_coeff = 0.0

                # ---- Horizontal direction ----
                if w is None and e is not None:
                    # Left boundary (x=0), normal points -x
                    # ∂p/∂x = qx, ghost: p_w = p_e - 2*dx*qx
                    diag_coeff += -2 * kx
                    values[nid, e] = 2 * kx
                    rhs_arr[nid] += -2 * qx / self._dx

                elif e is None and w is not None:
                    # Right boundary (x=lx), normal points +x
                    # ∂p/∂x = qx, ghost: p_e = p_w + 2*dx*qx
                    diag_coeff += -2 * kx
                    values[nid, w] = 2 * kx
                    rhs_arr[nid] += 2 * qx / self._dx

                elif e is not None and w is not None:
                    # Not a horizontal boundary
                    diag_coeff += -2 * kx
                    values[nid, e] = kx
                    values[nid, w] = kx

                # ---- Vertical direction ----
                if s is None and n is not None:
                    # Bottom boundary (y=0), normal points -y
                    # ∂p/∂y = qy, ghost: p_s = p_n - 2*dy*qy
                    diag_coeff += -2 * ky
                    values[nid, n] = 2 * ky
                    rhs_arr[nid] += -2 * qy / self._dy

                elif n is None and s is not None:
                    # Top boundary (y=ly), normal points +y
                    # ∂p/∂y = qy, ghost: p_n = p_s + 2*dy*qy
                    diag_coeff += -2 * ky
                    values[nid, s] = 2 * ky
                    rhs_arr[nid] += 2 * qy / self._dy

                elif n is not None and s is not None:
                    # Not a vertical boundary
                    diag_coeff += -2 * ky
                    values[nid, n] = ky
                    values[nid, s] = ky

                values[nid, nid] = diag_coeff

        # --------------------------------------------------
        # Assemble linear system
        # --------------------------------------------------
        matrix = self._create_matrix(field).from_data(values)

        # Build RHS field from array
        rhs_field = Field(field.mesh_shards, VariableType.SCALAR, field.etype)
        for nid in range(node_count):
            rhs_field[nid] = Variable.scalar(rhs_arr[nid])

        eqs = LinearEqs(matrix, rhs_field)
        return eqs

    @staticmethod
    def _extract_flux_components(flux) -> tuple[float, float]:
        """Extract (qx, qy) from a flux value (Variable, list, or scalar)."""
        qx, qy = 0.0, 0.0
        if isinstance(flux, Variable):
            data = flux.data
            if flux.vtype == VariableType.SCALAR:
                qx = qy = float(data)
            elif flux.vtype == VariableType.VECTOR:
                qx = float(data[0])
                qy = float(data[1])
        elif isinstance(flux, (list, tuple, np.ndarray)):
            arr = np.asarray(flux).flatten()
            qx = float(arr[0]) if len(arr) >= 1 else 0.0
            qy = float(arr[1]) if len(arr) >= 2 else 0.0
        elif np.isscalar(flux):
            qx = qy = float(flux)
        return qx, qy
