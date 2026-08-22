# -*- encoding: utf-8 -*-
"""
Copyright (C) 2026, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Burgers' equation solver using the finite difference method.
"""

from yunmeng.numerics.fields import Field, VariableType, FieldMeta, DataHub, Sample
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
from yunmeng.solvers.commons import supports
from yunmeng.setting import logger

import time
from dataclasses import dataclass


@dataclass
class BurgersConfig(SolverConfig):
    """Configuration for BurgersExplicitSolver."""

    time_order: int = 1  # Time integration order
    time_step: float = 0.001  # Initial time step
    end_time: float = 0.0  # End time
    min_dt: float = 1e-6  # Minimum time step
    cfl: float = 0.5  # CFL number
    nu: float = 0.01  # Kinematic viscosity

    @classmethod
    def get_solver_name(cls) -> str:
        return BurgersExplicitSolver.get_name()


class BurgersExplicitSolver(BaseSolver):
    """
    2D Burgers' equation explicit solver in fdm format on fixed 2d Grid.
    """

    @classmethod
    def get_meta(cls) -> SolverMeta:
        metas = SolverMeta()
        metas.description = "Fdm explicit solver for 2d Burgers equation"
        metas.type = SolverType.FDM
        metas.equation = "2d Burgers' equation"
        metas.equation_expr = "ddt(u) + grad(u)@u = lap(u, nu) + src(Q)"
        metas.dimension = MeshDimension.D2
        metas.default_ics = {"u": None}
        metas.default_bcs = {"u": None}
        metas.fields = {
            "u": FieldMeta(
                name="u",
                vtype=VariableType.vector(2),
                etype=ElementType.NODE,
            ),
        }
        return metas

    @classmethod
    def get_config_class(cls):
        return BurgersConfig

    @classmethod
    def get_name(cls) -> str:
        return "BurgersFdm2D"

    def __init__(self, id: str, mesh: Grid, operators: list[IOperator], configs: dict):
        super().__init__(id, mesh, operators, BurgersConfig.from_dict(configs))
        assert isinstance(mesh, Grid), "BurgersFdm2D only supports Grid."

        self._geom = mesh.get_geom_assistant()
        self._topo = mesh.get_topo_assistant()
        self._part = mesh.get_part_assistant()

        self._dx = None
        self._dy = None

        self._fields = {
            "u": Field(
                self._part.shards, VariableType.vector(2), ElementType.NODE, name="u"
            )
        }
        self._buffs: DataHub = None

    def initialize(self):
        # Check initial conditions
        if "u" not in self._ics:
            raise ValueError(f"Solver {self._id} has no initial condition for u.")

        self._ics["u"].apply(self._fields["u"])

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

        # Init configs
        self._time_step = self._config.time_step
        self._cfl = self._config.cfl
        self._dx = self._mesh.lx / self._mesh.nx
        self._dy = self._mesh.ly / self._mesh.ny

        # Init operators
        for op in self._operators:
            op.prepare(self._mesh, bounds=self._bcs)

        # Init buffers
        time_order = 2
        self._buffs = DataHub(["u"], time_order)
        for _ in range(time_order):
            self._buffs.push("u", Sample(0.0, self._fields["u"]), ElementType.NODE)

        # Call callbacks
        for callback in self._callbacks:
            callback.on_task_begin()

    def step(self) -> SolverStatus:
        start = time.perf_counter()

        # Compute time step
        curr_time = self._status.current_time
        dt = supports.cfl_timestep(
            self._mesh, self._fields["u"], self._cfl, min_dt=1e-3
        )
        rest_time = self._status.end_time - curr_time
        dt = min(dt, self._time_step, rest_time)

        # Call callbacks
        for callback in self._callbacks:
            callback.on_step_begin()

        # Update solution
        old_u = self._fields["u"]
        u_grad, u_diff, u_src = None, None, None
        for op in self._operators:
            if op.get_type() == OperatorType.GRAD:
                u_grad = op(self._buffs, curr_time)
            elif op.get_type() == OperatorType.LAPLACIAN:
                u_diff = op(self._buffs, curr_time)
            elif op.get_type() == OperatorType.SRC:
                u_src = op(self._buffs, curr_time)

        u_conv = u_grad @ old_u
        new_u = old_u - dt * u_conv + dt * u_diff + dt * u_src

        # Update status
        time_cost = time.perf_counter() - start
        self._update_status(time_cost, dt)

        self._fields["u"] = new_u
        self._apply_boundary_conditions()
        self._buffs.push(
            "u", Sample(self._status.current_time, new_u), ElementType.NODE
        )

        # Call callbacks
        for callback in self._callbacks:
            callback.on_step_end()

        if self._status.finished:
            for callback in self._callbacks:
                callback.on_task_end()

        return self._status

    def _apply_boundary_conditions(self):
        """Apply boundary conditions to the velocity field."""
        for target_field, bcs in self._bcs.items():
            for bc in bcs:
                if bc.get_type() == BoundaryType.VALUE:
                    bc.apply(self._fields[target_field])

    def _update_status(self, time_cost: float, dt: float):
        self._status.current_time += dt
        self._status.step_time = time_cost
        self._status.total_time += time_cost
        self._status.time_step = dt

        if self._status.current_time >= self._status.end_time:
            self._status.finished = True
