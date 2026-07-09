# -*- encoding: utf-8 -*-
"""
Copyright (C) 2026, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Navier-Stokes equations solver in fdm format on fixed 2d Grid.
"""

from yunmeng.numerics.fields import Field, VariableType, FieldMeta, DataHub2, Sample2
from yunmeng.numerics.grids import Grid, ElementType, MeshDimension
from yunmeng.solvers.commons import (
    BaseSolver,
    SolverMeta,
    SolverStatus,
    SolverType,
    IOperator,
    OperatorType,
)
from yunmeng.solvers.interfaces import SolverConfig, BoundaryType
from yunmeng.setting import logger

import time
from dataclasses import dataclass


@dataclass
class NSConfig(SolverConfig):
    """Configuration for NavierStokesSolver."""

    nu: float = 0.01  # Kinematic viscosity
    rho: float = 1.0  # Fluid density (was hard-coded)

    time_order: int = 1  # Time integration order
    time_step: float = 0.001  # Initial time step
    end_time: float = 0.0  # End time
    cfl: float = 0.5  # CFL number
    min_dt: float = 1e-6  # Minimum allowed time step

    @classmethod
    def get_solver_name(cls) -> str:
        return NavierStokesSolver.get_name()


class NavierStokesSolver(BaseSolver):
    """
    2D Navier-Stokes equations solver by Chorin's Projection method.

    First, solve the momentum equation to update u_star:
      ddt(u) + u*grad(u) = lap(u; nu)
    Second, solve the pressure Poisson equation to update p:
      lap(p) = (RHO/dt)*div(u_star)
    Finally, update u_new:
      ddt(u_star) = - grad(p; 1/RHO)
    """

    @classmethod
    def get_meta(cls) -> SolverMeta:
        metas = SolverMeta()
        metas.description = "Fdm explicit solver for 2d Navier-Stokes equations"
        metas.type = SolverType.FDM
        metas.equation = "2d Navier-Stokes equation"
        metas.equation_expr = "ddt(u) + grad(u)@u = lap(u; nu) - grad(p; 1/RHO)"
        metas.dimension = MeshDimension.D2
        metas.default_ics = {"u": None, "p": None}
        metas.default_bcs = {"u": None, "p": None}
        metas.fields = {
            "u": FieldMeta(
                name="u",
                vtype=VariableType.vector(2),
                etype=ElementType.NODE,
            ),
            "p": FieldMeta(
                name="p",
                vtype=VariableType.scalar(),
                etype=ElementType.NODE,
            ),
        }
        return metas

    @classmethod
    def get_config_class(cls):
        return NSConfig

    @classmethod
    def get_name(cls) -> str:
        return "NavierStokesFdm2D"

    def __init__(self, id: str, mesh: Grid, operators: list[IOperator], configs: dict):
        super().__init__(id, mesh, operators, NSConfig.from_dict(configs))
        assert isinstance(mesh, Grid), f"{self.get_name()} only supports Grid."

        self._geom = mesh.get_geom_assistant()
        self._topo = mesh.get_topo_assistant()
        self._part = mesh.get_part_assistant()

        self._dx = None
        self._dy = None

        self._fields = {
            "u": Field(
                self._part.shards, VariableType.vector(2), ElementType.NODE, name="u"
            ),
            "p": Field(
                self._part.shards, VariableType.scalar(), ElementType.NODE, name="p"
            ),
        }
        self._buffs: DataHub2 = None

    def initialize(self):
        # Check initial conditions
        if "u" not in self._ics:
            raise ValueError(f"Solver {self._id} has no initial condition for u.")
        if "p" not in self._ics:
            raise ValueError(f"Solver {self._id} has no initial condition for p.")

        for field in self._fields.keys():
            self._ics[field].apply(self._fields[field])

        # Check boundary conditions
        existed_ids = []
        for bc in self._bcs["u"]:
            existed_ids.extend(bc.region.get_element_ids())
        if len(existed_ids) != self._topo.boundary_nodes.size:
            boundary_nodes = self._topo.boundary_nodes
            missed_ids = set(boundary_nodes) - set(existed_ids)
            raise ValueError(
                f"Solver {self._id} boundary condition for u is not complete. "
                f"Missed node ids: {missed_ids}."
            )

        # Init status
        self._status = SolverStatus()
        self._status.time_step = self._config.time_step
        self._status.end_time = self._config.end_time
        self._status.finished = False
        self._status.current_time = 0.0
        self._status.total_time = 0.0
        self._status.steps = 0

        # Init configs
        self._time_step = self._config.time_step
        self._cfl = self._config.cfl
        self._rho = self._config.rho
        self._dx = self._mesh.lx / self._mesh.nx
        self._dy = self._mesh.ly / self._mesh.ny

        # Init operators
        for op in self._operators:
            op.prepare(self._mesh, bounds=self._bcs)

        # Init buffers
        time_order = 2
        self._buffs = DataHub2(["u", "p"], time_order)
        for _ in range(time_order):
            self._buffs.push("u", Sample2(0.0, self._fields["u"]), ElementType.NODE)
            self._buffs.push("p", Sample2(0.0, self._fields["p"]), ElementType.NODE)

        # Call callbacks
        for callback in self._callbacks:
            callback.on_task_begin()

    def forward(self) -> SolverStatus:
        start = time.perf_counter()

        # Compute time step
        curr_time = self._status.current_time
        rest_time = self._status.end_time - curr_time
        if rest_time <= 1e-6:
            self._status.finished = True
            for callback in self._callbacks:
                callback.on_task_end()
            return self._status

        dt = min(self._time_step, rest_time)
        new_t = curr_time + dt

        # Call callbacks
        for callback in self._callbacks:
            callback.on_step_begin()

        # Update solution of u_star
        self._solve_momentum(dt, new_t)

        # Apply boundary conditions
        self._apply_boundary_conditions()

        # Update solution of p
        self._solve_pressure(dt, new_t)

        # Correct solution of u_new
        self._correct_velocity(dt, new_t)

        # Apply boundary conditions
        self._apply_boundary_conditions()

        # Update status
        time_cost = time.perf_counter() - start
        self._update_status(time_cost, dt)

        # Call callbacks
        for callback in self._callbacks:
            callback.on_step_end()

        if self._status.finished:
            for callback in self._callbacks:
                callback.on_task_end()

        return self._status

    def _apply_boundary_conditions(self):
        """Apply boundary conditions to the fields."""
        for target_field, bcs in self._bcs.items():
            for bc in bcs:
                if bc.get_type() == BoundaryType.VALUE:
                    bc.apply(self._fields[target_field])

    def _solve_momentum(self, dt: float, new_t: float):
        """Solve the momentum equation to get tentative velocity."""
        u = self._fields["u"]
        u_grad, u_diff = None, None
        for op in self._operators:
            if op.get_type() == OperatorType.GRAD and "u" in op.target_fields:
                u_grad = op(self._buffs, new_t)
            elif op.get_type() == OperatorType.LAPLACIAN and "u" in op.target_fields:
                u_diff = op(self._buffs, new_t)

        u_conv = u_grad @ u
        u_star = u - dt * u_conv + dt * u_diff
        for bc in self._bcs["u"]:
            bc.apply(u_star)

        self._fields["u"] = u_star
        self._buffs.push("u", Sample2(new_t, u_star), ElementType.NODE)

    def _solve_pressure(self, dt: float, new_t: float):
        """Solve the poisson equation to get pressure."""
        u_div, p_eqs = None, None
        for op in self._operators:
            if op.get_type() == OperatorType.DIV:
                u_div = op(self._buffs, new_t)
            elif op.get_type() == OperatorType.LAPLACIAN and "p" in op.target_fields:
                p_eqs = op(self._buffs, new_t)

        # only apply divergence on internal nodes
        rhs = p_eqs.rhs.copy()
        for nid in self._topo.internal_nodes:
            div_val = u_div[nid]
            rhs[nid] = (self._rho / dt) * div_val

        p_eqs.reset_rhs(rhs)
        new_p = p_eqs.solve()
        self._fields["p"] = new_p

        # update pressure field
        self._buffs.push("p", Sample2(new_t, new_p), ElementType.NODE)

    def _correct_velocity(self, dt: float, new_t: float):
        """Correct the velocity field by pressure for continuty."""
        u = self._fields["u"]
        p_grad = None
        for op in self._operators:
            if op.get_type() == OperatorType.GRAD and "p" in op.target_fields:
                p_grad = op(self._buffs, new_t)

        new_u = u - (dt / self._rho) * p_grad
        self._fields["u"] = new_u

        # update velocity
        self._buffs.push("u", Sample2(new_t, new_u), ElementType.NODE)

    def _update_status(self, time_cost: float, dt: float):
        self._status.current_time += dt
        self._status.step_time = time_cost
        self._status.total_time += time_cost
        self._status.time_step = dt
        self._status.steps += 1

        if self._status.current_time >= self._status.end_time:
            self._status.finished = True
