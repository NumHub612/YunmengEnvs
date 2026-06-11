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
from yunmeng.numerics.mesh import Grid2D, ElementType
from yunmeng.numerics.algos.topos import MeshTopo
from yunmeng.numerics.mats.linalgs import LinearEqs, Matrix
from yunmeng.numerics.mats.sparse import NumpyMatrix, TorchMatrix
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

        self._mesh: Grid2D = None
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
        mesh: Grid2D,
        bounds: dict[int, dict[str, IBoundaryCondition]],
    ):
        if not isinstance(mesh, Grid2D):
            raise ValueError("FDM op lap01 only supports Grid2D.")
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

        self._mesh: Grid2D = None
        self._topo: MeshTopo = None
        self._bcs = None

        self._dx = None
        self._dy = None

    @property
    def target_fields(self) -> list[str]:
        return [self._var]

    def prepare(
        self, mesh: Grid2D, bounds: dict[int, dict[str, IBoundaryCondition]] = None
    ):
        if not isinstance(mesh, Grid2D):
            raise ValueError("FDM op lap02 only supports Grid2D.")
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
        rhs = Field.from_shard(field.field_shards, field.mesh_shards, field.meta)

        kx = 1.0 / self._dx**2
        ky = 1.0 / self._dy**2

        for nid in self._topo.internal_nodes:
            # Neighbour nodes
            e, w, n, s, _, _ = self._mesh.get_node_neighbours(nid)
            values[nid, nid] = -2 * (kx + ky)

            # Horizontal
            values[nid, e] = kx
            values[nid, w] = kx

            # Vertical
            values[nid, n] = ky
            values[nid, s] = ky

        for nid in self._topo.boundary_nodes:
            # Neighbour nodes
            e, w, n, s, _, _ = self._mesh.get_node_neighbours(nid)
            values[nid, nid] = -2 * (kx + ky)

            # Horizontal result
            if e is None:
                values[nid, nid] += kx
            else:
                values[nid, e] += kx

            if w is None:
                values[nid, nid] += kx
            else:
                values[nid, w] += kx

            # Vertical result
            if n is None:
                values[nid, nid] += ky
            else:
                values[nid, n] += ky

            if s is None:
                values[nid, nid] += ky
            else:
                values[nid, s] += ky

        # Boundary conditions
        for nid in self._topo.boundary_nodes:
            bc = self._bcs[nid][self._var]
            if bc.get_type() == BoundaryType.VALUE:
                value = bc.evaluate().value
                for i in range(node_count):
                    values[nid, i] = 0.0
                values[nid, nid] = 1.0
                rhs[nid] = value

        matrix = self._create_matrix(field).from_data(values)
        eqs = LinearEqs(matrix, rhs)
        return eqs
