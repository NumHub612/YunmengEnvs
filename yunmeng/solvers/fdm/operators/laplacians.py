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
from yunmeng.numerics.mesh import Grid2D, ElementType
from yunmeng.numerics.algos.topos import MeshTopo
from yunmeng.numerics.fields import DataHub, Field, Variable, VariableType


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

    def __init__(self, diffusivity: float = 1.0):
        self._mesh: Grid2D = None
        self._topo: MeshTopo = None

        self._bcs = None
        self._var = ""
        self._nu = diffusivity
        self._dx = None
        self._dy = None

    def prepare(
        self,
        fields: list[str],
        mesh: Grid2D,
        bounds: dict[int, dict[str, IBoundaryCondition]],
    ):
        if not isinstance(mesh, Grid2D):
            raise ValueError("FDM op lap01 only supports Grid2D.")
        if not mesh.orthogonal:
            # TODO: Support non-orthogonal grids
            raise ValueError("FDM op lap01 requires orthogonal grids.")
        if len(fields) != 1:
            raise ValueError("FDM op lap01 only supports one field.")
        for bc in bounds.values():
            # TODO: Support more types of boundary conditions
            for v in bc.values():
                if v.get_type() != BoundaryType.VALUE:
                    raise ValueError("FDM op lap01 requires value BC.")

        self._mesh = mesh
        self._bcs = bounds
        self._var = fields[0]

        self._topo = self._mesh.get_topo_assistant()
        self._dx = self._mesh.lx / (self._mesh.nx - 1)
        self._dy = self._mesh.ly / (self._mesh.ny - 1)

    def run(self, sources: DataHub, timestep: float = None) -> Field:
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
