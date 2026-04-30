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
from yunmeng.numerics.mesh import Grid, ElementType
from yunmeng.numerics.algos.topos import MeshTopo
from yunmeng.numerics.fields import DataHub, Field, Var


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
        self._mesh: Grid = None
        self._topo: MeshTopo = None

        self._bcs = None
        self._var = ""
        self._nu = diffusivity
        self._dx = None
        self._dy = None

    def prepare(
        self,
        fields: list[str],
        mesh: Grid,
        bounds: dict[int, dict[str, IBoundaryCondition]],
    ):
        if not isinstance(mesh, Grid):
            raise ValueError("FDM op lap01 only supports Grid.")
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

    def run(self, sources: DataHub, timestep: float) -> Field:
        old_field = sources.field(self._var, loc=ElementType.NODE).data
        new_field = old_field.copy()
        if len(old_field.mesh_shards) != 1:
            # TODO: Support multi-gpu divergence operator
            raise ValueError("FDM op lap01 only supports cpu.")

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
        """Update internal nodes using the explicit scheme."""
        new_field = field.copy()
        kx = self._nu * dt / self._dx**2
        ky = self._nu * dt / self._dy**2

        for nid in self._topo.internal_nodes:
            # Neighbour nodes
            e, w, n, s, _, _ = self._mesh.get_node_neighbours(nid)
            ue, uw, un, us, ui = field[[e, w, n, s, nid]]

            # Horizontal flux
            uh = ue - 2 * ui + uw

            # Vertical flux
            uv = un - 2 * ui + us

            # Total flux
            new_u = ui + kx * uh + ky * uv
            new_field[nid] = new_u
        return new_field
