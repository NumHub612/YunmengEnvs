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
from yunmeng.numerics.mesh import Grid2D, ElementType
from yunmeng.numerics.algos.topos import MeshTopo
from yunmeng.numerics.fields import DataHub, Field, Variable, VariableType


class Grad01(IOperator):
    """
    First order upwind explicit gradient operator on structured grids.
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

        self._mesh: Grid2D = None
        self._topo: MeshTopo = None
        self._bcs = None

        self._dx = None
        self._dy = None

    def prepare(
        self,
        mesh: Grid2D,
        bounds: dict[int, dict[str, IBoundaryCondition]],
    ):
        if not isinstance(mesh, Grid2D):
            raise ValueError("FDM op grad01 only supports Grid2D.")
        if not mesh.orthogonal:
            # TODO: Support non-orthogonal grids
            raise ValueError("FDM op grad01 requires orthogonal grids.")
        for bc in bounds.values():
            # no need to calculate gradient at boundary if Dirichlet BC.
            # TODO: Support more types of boundary conditions
            for v in bc.values():
                if v.get_type() != BoundaryType.VALUE:
                    raise ValueError("FDM op grad01 requires value BC.")

        self._mesh = mesh
        self._bcs = bounds

        self._topo = self._mesh.get_topo_assistant()
        self._dx = self._mesh.lx / (self._mesh.nx - 1)
        self._dy = self._mesh.ly / (self._mesh.ny - 1)

    def run(self, sources: DataHub, timestep: float = None) -> Field:
        """Calculate the gradient of the field."""
        old_field = sources.field(self._var, loc=ElementType.NODE).data
        new_field = old_field.copy()
        if len(old_field.mesh_shards) != 1:
            # TODO: Support multi-gpu gradient operator
            raise ValueError("FDM op grad01 only supports cpu.")

        # Apply boundary conditions
        self._apply_bc(new_field)

        # Calculate the gradient
        if old_field.vtype == VariableType.SCALAR:
            grads = self._calculate_scalar_field(new_field)
        elif old_field.vtype == VariableType.VECTOR:
            grads = self._calculate_vector_field(new_field)
        else:
            raise ValueError("FDM op grad01 not support tensor fields.")

        return grads

    def _apply_bc(self, field: Field):
        """Apply boundary conditions to the field."""
        for nid in self._topo.boundary_nodes:
            bc = self._bcs[nid][self._var]
            value = bc.evaluate().value
            field[nid] = value

    def _check_upwind(self, c: float) -> tuple[float, float]:
        """Check if the flux is upwind or downwind."""
        return (max(c / (abs(c) + 1e-6), 0), max(-c / (abs(c) + 1e-6), 0))

    def _calculate_vector_field(self, field: Field) -> Field:
        """Calculate the gradient of the vector field."""
        new_field = Field(field.mesh_shards, VariableType.TENSOR, field.etype)
        kx = 1 / self._dx
        ky = 1 / self._dy

        for nid in self._topo.internal_nodes:
            # Neighbour nodes
            e, w, n, s, _, _ = self._mesh.get_node_neighbours(nid)
            ui = field[nid]

            # Horizontal upwind
            fh1, fh2 = self._check_upwind(ui[0])
            ue = ui * fh1 + field[e] * fh2
            uw = field[w] * fh1 + ui * fh2

            # Vertical upwind
            fv1, fv2 = self._check_upwind(ui[1])
            un = ui * fv1 + field[n] * fv2
            us = field[s] * fv1 + ui * fv2

            # Gradients of u
            ux = (ue[0] - uw[0]) * kx
            uy = (un[0] - us[0]) * ky

            # Gradients of v
            vx = (ue[1] - uw[1]) * kx
            vy = (un[1] - us[1]) * ky

            # Total gradient
            grad = Variable.tensor(ux, vx, uy, vy)
            new_field[nid] = grad

        return new_field

    def _calculate_scalar_field(self, field: Field) -> Field:
        """Calculate the gradient of the scalar field."""
        new_field = Field(field.mesh_shards, VariableType.VECTOR, field.etype)
        kx = 1 / self._dx
        ky = 1 / self._dy
        for nid in self._topo.internal_nodes:
            # Neighbour nodes
            e, w, n, s, _, _ = self._mesh.get_node_neighbours(nid)
            u = field[nid]

            # Horizontal flux
            fh1, fh2 = self._check_upwind(u)
            ue = u * fh1 + field[e] * fh2
            uw = field[w] * fh1 + u * fh2

            # Vertical flux
            fv1, fv2 = self._check_upwind(u)
            un = u * fv1 + field[n] * fv2
            us = field[s] * fv1 + u * fv2

            # Gradient
            ux = (ue - uw) * kx
            uy = (un - us) * ky
            grad = Variable.vector(ux, uy)
            new_field[nid] = grad

        return new_field
