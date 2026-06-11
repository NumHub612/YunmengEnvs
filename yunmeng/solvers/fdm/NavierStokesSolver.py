# -*- encoding: utf-8 -*-
"""
Copyright (C) 2026, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Navier-Stokes equations solver in fdm format on fixed 2d Grid.
"""

from yunmeng.numerics.fields import Field, VariableType, FieldMeta, DataHub, Sample
from yunmeng.numerics.mesh import Grid2D, ElementType, MeshDimension
from yunmeng.solvers.commons import (
    BaseSolver,
    SolverMeta,
    SolverStatus,
    SolverType,
    IOperator,
    OperatorType,
)
from yunmeng.solvers.interfaces import BoundaryType
from yunmeng.solvers.commons import inits, boundaries, supports
from yunmeng.numerics.consts import RHO
from yunmeng.setting import logger

import time


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
        metas.default_ics = {
            "u": inits.UniformInitialization,
            "p": inits.UniformInitialization,
        }
        metas.default_bcs = {
            "u": boundaries.WallBoundary,
            "p": boundaries.WallBoundary,
        }
        metas.fields = {
            "u": FieldMeta(
                vtype=VariableType.VECTOR,
                etype=ElementType.NODE,
            ),
            "p": FieldMeta(
                vtype=VariableType.SCALAR,
                etype=ElementType.NODE,
            ),
        }
        return metas

    @classmethod
    def get_name(cls) -> str:
        return "NavierStokesFdm2D"

    def __init__(self, id: str, mesh: Grid2D, operators: list[IOperator]):
        super().__init__(id, mesh, operators)
        assert isinstance(mesh, Grid2D), f"{self.get_name()} only supports Grid2D."

        self._geom = mesh.get_geom_assistant()
        self._topo = mesh.get_topo_assistant()
        self._part = mesh.get_part_assistant()

        self._time_step = 0.001
        self._cfl = 0.5
        self._nu = 0.01
        self._cfl = 0.5
        self._dx = None
        self._dy = None

        self._default_ics = {
            "u": inits.UniformInitialization("u", [0.0, 0.0, 0.0]),
            "p": inits.UniformInitialization("p", 0.0),
        }
        self._default_bcs = {
            "u": boundaries.WallBoundary("u"),
            "p": boundaries.WallBoundary("p"),
        }

        self._fields = {
            "u": Field(self._part.shards, VariableType.VECTOR, ElementType.NODE),
            "p": Field(self._part.shards, VariableType.SCALAR, ElementType.NODE),
        }
        self._buffs: DataHub = None

    def initialize(
        self,
        total_time: float,
        time_step: float,
        cfl: float = 0.5,
    ):
        """
        Initialize the solver.

        Args:
            total_time: The total time of the simulation.
            time_step: The time step for the simulation.
            cfl: The CFL number for time step calculation.
        """

        # Check initial conditions
        if "u" not in self._ics:
            self._ics["u"] = self._default_ics["u"]
            logger.warning(
                f"Solver {self._id} has no initial condition for u, using default ic."
            )
        if "p" not in self._ics:
            self._ics["p"] = self._default_ics["p"]
            logger.warning(
                f"Solver {self._id} has no initial condition for p, using default ic."
            )

        for field in self._fields.keys():
            self._ics[field].apply(self._fields[field])

        # Check boundary conditions
        for node in self._topo.boundary_nodes:
            for field in self._fields.keys():
                if node not in self._bcs or field not in self._bcs[node]:
                    self._bcs[node][field] = self._default_bcs[field]
                    logger.warning(
                        f"Solver {self._id} has no boundary condition for {field} "
                        f"at node {node}, using default bc."
                    )

        # Init status
        self._status = SolverStatus()
        self._status.time_step = time_step
        self._status.end_time = total_time
        self._status.finished = False
        self._status.current_time = 0.0
        self._status.total_time = 0.0
        self._status.steps = 0

        # Init configs
        self._time_step = time_step
        self._cfl = cfl
        self._dx = self._mesh.lx / self._mesh.nx
        self._dy = self._mesh.ly / self._mesh.ny

        # Init operators
        for op in self._operators:
            op.prepare(self._mesh, bounds=self._bcs)

        # Init buffers
        time_order = 2
        self._buffs = DataHub(["u", "p"], time_order)
        for _ in range(time_order):
            self._buffs.push_field("u", Sample(0.0, self._fields["u"]))
            self._buffs.push_field("p", Sample(0.0, self._fields["p"]))

        # Call callbacks
        for callback in self._callbacks:
            callback.on_task_begin()

    def inference(self) -> SolverStatus:
        start = time.perf_counter()

        # Compute time step
        dt = supports.cfl_timestep(
            self._mesh, self._fields["u"], self._cfl, min_dt=1e-3
        )
        rest_time = self._status.end_time - self._status.current_time
        dt = min(dt, self._time_step, rest_time)
        new_t = self._status.current_time + dt

        # Call callbacks
        for callback in self._callbacks:
            callback.on_step_begin()

        # Apply boundary conditions
        self._apply_boundary_conditions()

        # Update solution of u_star
        self._solve_momentum(dt, new_t)

        # Update solution of p
        self._solve_pressure(dt, new_t)

        # Correct solution of u_new
        self._correct_velocity(dt, new_t)

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
        for nid in self._topo.boundary_nodes:
            for var in self._fields.keys():
                bc = self._bcs[nid][var]
                if bc.get_type() == BoundaryType.VALUE:
                    value = bc.evaluate().value
                    self._fields[var][nid] = value

    def _solve_momentum(self, dt: float, new_t: float):
        """Solve the momentum equation to get tentative velocity."""
        u = self._fields["u"]
        u_grad, u_diff = None, None
        for op in self._operators:
            if op.get_type() == OperatorType.GRAD and "u" in op.target_fields:
                u_grad = op.run(u, dt)
            elif op.get_type() == OperatorType.LAPLACIAN and "u" in op.target_fields:
                u_diff = op.run(u, dt)

        u_star = u - dt * u_grad @ u + dt * u_diff
        self._fields["u"] = u_star

        # update velocity and gradient temporarly
        self._buffs.push("u", Sample(new_t, u_star), Sample(new_t, u_grad))

    def _solve_pressure(self, dt: float, new_t: float):
        """Solve the poisson equation to get pressure."""
        u_div, p_eqs = None, None
        for op in self._operators:
            if op.get_type() == OperatorType.DIV:
                u_div = op.run(self._buffs, dt)
            elif op.get_type() == OperatorType.LAPLACIAN and "p" in op.target_fields:
                p_eqs = op.run(self._buffs, dt)

        rhs = p_eqs.rhs + (1 / dt) * u_div
        p_eqs.reset_rhs(rhs)
        new_p = p_eqs.solve()
        self._fields["p"] = new_p

        # update pressure field
        self._buffs.push_field("p", Sample(new_t, new_p))

    def _correct_velocity(self, dt: float, new_t: float):
        """Correct the velocity field by pressure for continuty."""
        u = self._fields["u"]
        p_grad = None
        for op in self._operators:
            if op.get_type() == OperatorType.GRAD and "p" in op.target_fields:
                p_grad = op.run(self._buffs, dt)

        new_u = u - (dt / 1) * p_grad
        self._fields["u"] = new_u

        # update velocity
        self._buffs.push_field("u", Sample(new_t, new_u))
        self._buffs.push_grad("p", Sample(new_t, p_grad))

    def _update_status(self, time_cost: float, dt: float):
        self._status.current_time += dt
        self._status.step_time = time_cost
        self._status.total_time += time_cost
        self._status.time_step = dt
        self._status.steps += 1

        if self._status.current_time >= self._status.end_time:
            self._status.finished = True
