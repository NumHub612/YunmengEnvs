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
            # TODO: Support non-uniform grids
            raise ValueError(f"FDM op {self.get_name()} requires uniform grids.")
        for bc in bounds.values():
            # no need to calculate gradient at boundary if Dirichlet BC.
            # TODO: Support more types of boundary conditions
            for fname, v in bc.items():
                if fname == self._var and v.get_type() != BoundaryType.VALUE:
                    raise ValueError(f"FDM op {self.get_name()} requires value BC.")

        self._mesh = mesh
        self._bcs = bounds

        self._topo = self._mesh.get_topo_assistant()
        self._dx = self._mesh.lx / (self._mesh.nx - 1)
        self._dy = self._mesh.ly / (self._mesh.ny - 1)

    def run(self, sources: Field | DataHub, dt: float = None) -> Field:
        """Calculate the gradient of the field."""
        if isinstance(sources, Field):
            old_field = sources
        else:
            old_field = sources.field(self._var, loc=ElementType.NODE).data
        new_field = old_field.copy()
        if len(old_field.mesh_shards) != 1:
            # TODO: Support multi-gpu gradient operator
            raise ValueError(f"FDM op {self.get_name()} only supports cpu.")

        # Apply boundary conditions
        self._apply_bc(new_field)

        # Calculate the gradient
        if old_field.vtype == VariableType.SCALAR:
            grads = self._calculate_scalar_field(new_field)
        elif old_field.vtype == VariableType.VECTOR:
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

    def _check_upwind(self, c: float) -> tuple[float, float]:
        """Check if the flux is upwind or downwind."""
        if abs(c) < 1e-6:  # Back to central difference
            return (0.5, 0.5)
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
            grad = Variable.tensor([ux, uy, vx, vy])
            new_field[nid] = grad

        for nid in self._topo.boundary_nodes:
            # Neighbour nodes
            e, w, n, s, _, _ = self._mesh.get_node_neighbours(nid)
            ui = field[nid]

            # horizontal flux
            if e and w:
                fh1, fh2 = self._check_upwind(ui[0])
                ue = ui * fh1 + field[e] * fh2
                uw = field[w] * fh1 + ui * fh2
            else:
                ue = field[e] if e else ui
                uw = field[w] if w else ui

            # vertical flux
            if n and s:
                fv1, fv2 = self._check_upwind(ui[1])
                un = ui * fv1 + field[n] * fv2
                us = field[s] * fv1 + ui * fv2
            else:
                un = field[n] if n else ui
                us = field[s] if s else ui

            # Gradient
            ux = (ue[0] - uw[0]) * kx
            uy = (un[0] - us[0]) * ky
            vx = (ue[1] - uw[1]) * kx
            vy = (un[1] - us[1]) * ky
            grad = Variable.tensor([ux, uy, vx, vy])
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


class Grad02(IOperator):
    """
    Second order central explicit gradient operator on structured grids.

    Technical features:
    + The central difference scheme utilizes symmetric neighboring node information
    to achieve second-order without artificial numerical dissipation,
    accurately capturing flow field details.

    Use when:
    +  It is mainly applied to discretize the pressure Poisson equation,
    viscous diffusion terms, and low Reynolds number flows,
    meeting strict requirements for global conservation and accuracy.
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
            # TODO: Support non-uniform grids
            raise ValueError(f"FDM op {self.get_name()} requires uniform grids.")

        self._mesh = mesh
        self._bcs = bounds

        self._topo = self._mesh.get_topo_assistant()
        self._dx = self._mesh.lx / (self._mesh.nx - 1)
        self._dy = self._mesh.ly / (self._mesh.ny - 1)

    def run(self, sources: Field | DataHub, dt: float = None) -> Field:
        """Calculate the gradient of the field."""
        if isinstance(sources, Field):
            old_field = sources
        else:
            old_field = sources.field(self._var, loc=ElementType.NODE).data
        new_field = old_field.copy()
        if len(old_field.mesh_shards) != 1:
            # TODO: Support multi-gpu gradient operator
            raise ValueError(f"FDM op {self.get_name()} only supports cpu.")

        # Apply boundary conditions
        self._apply_bc(new_field)

        # Calculate the gradient
        if old_field.vtype == VariableType.SCALAR:
            grads = self._calculate_scalar_field(new_field)
        elif old_field.vtype == VariableType.VECTOR:
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
        """Calculate the gradient of the vector field."""
        new_field = Field(field.mesh_shards, VariableType.TENSOR, field.etype)
        kx = 1.0 / (2.0 * self._dx)
        ky = 1.0 / (2.0 * self._dy)

        for nid in self._topo.internal_nodes:
            # Neighbour nodes
            e, w, n, s, _, _ = self._mesh.get_node_neighbours(nid)
            ue, uw, un, us = field[[e, w, n, s]]

            # Gradients of u
            ux = (ue[0] - uw[0]) * kx
            uy = (un[0] - us[0]) * ky

            # Gradients of v
            vx = (ue[1] - uw[1]) * kx
            vy = (un[1] - us[1]) * ky

            new_field[nid] = Variable.tensor([ux, uy, vx, vy])

        for nid in self._topo.boundary_nodes:
            bc = self._bcs[nid][self._var]
            e, w, n, s, _, _ = self._mesh.get_node_neighbours(nid)
            ue, uw, un, us = field[[e, w, n, s]]

            if bc.get_type() == BoundaryType.FLUX:
                flux = bc.evaluate().flux
                new_field[nid] = flux
            else:
                if e and w:
                    ux = (ue[0] - uw[0]) * kx
                    vx = (ue[1] - uw[1]) * kx
                else:
                    ue = field[e] if e else field[w]
                    uw = field[w] if w else field[e]
                    ux = 2.0 * (ue[0] - uw[0]) * kx

                if n and s:
                    uy = (un[0] - us[0]) * ky
                    vy = (un[1] - us[1]) * ky
                else:
                    un = field[n] if n else field[s]
                    us = field[s] if s else field[n]
                    uy = 2.0 * (un[0] - us[0]) * ky

                new_field[nid] = Variable.tensor([ux, uy, vx, vy])

        return new_field

    def _calculate_scalar_field(self, field: Field) -> Field:
        """Calculate the gradient of the scalar field."""
        new_field = Field(field.mesh_shards, VariableType.VECTOR, field.etype)
        kx = 1.0 / (2.0 * self._dx)
        ky = 1.0 / (2.0 * self._dy)

        for nid in self._topo.internal_nodes:
            # Neighbour nodes
            e, w, n, s, _, _ = self._mesh.get_node_neighbours(nid)

            # Centeral difference
            ux = (field[e] - field[w]) * kx
            uy = (field[n] - field[s]) * ky

            new_field[nid] = Variable.vector(ux, uy)

        for nid in self._topo.boundary_nodes:
            bc = self._bcs[nid][self._var]
            if bc.get_type() == BoundaryType.FLUX:
                value = bc.evaluate().flux
                new_field[nid] = value
        return new_field
