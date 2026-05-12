# -*- encoding: utf-8 -*-
"""
Copyright (C) 2026, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Burgers' equation solver using the finite difference method.
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
from yunmeng.solvers.commons import inits, boundaries, supports
from yunmeng.setting import logger

import time


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
        metas.equation_expr = "ddt(u) + u*grad(u) = lap(u, nu) + src(Q)"
        metas.dimension = MeshDimension.D2
        metas.default_ics = {"u": inits.UniformInitialization}
        metas.default_bcs = {"u": boundaries.WallBoundary}
        metas.fields = {
            "u": FieldMeta(
                vtype=VariableType.VECTOR,
                etype=ElementType.NODE,
            ),
        }
        return metas

    @classmethod
    def get_name(cls) -> str:
        return "BurgersFdm2D"

    def __init__(self, id: str, mesh: Grid2D, operators: list[IOperator]):
        super().__init__(id, mesh, operators)
        assert isinstance(mesh, Grid2D), "BurgersFdm2D only supports Grid2D."

        self._geom = mesh.get_geom_assistant()
        self._topo = mesh.get_topo_assistant()
        self._part = mesh.get_part_assistant()

        self._time_step = 0.001
        self._cfl = 0.5
        self._nu = 0.01
        self._cfl = 0.5
        self._dx = None
        self._dy = None

        self._default_bcs = {"u": boundaries.WallBoundary("u")}
        self._default_ics = {"u": inits.UniformInitialization("u", [0.0, 0, 0])}

        self._fields = {
            "u": Field(self._part.shards, VariableType.VECTOR, ElementType.NODE)
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

        self._ics["u"].apply(self._fields["u"])

        # Check boundary conditions
        for node in self._topo.boundary_nodes:
            if node not in self._bcs or "u" not in self._bcs[node]:
                self._bcs[node]["u"] = self._default_bcs["u"]
                logger.warning(
                    f"Solver {self._id} has no boundary condition for u at node {node}, "
                    f"using default bc."
                )

        # Init status
        self._status = SolverStatus()
        self._status.time_step = time_step
        self._status.end_time = total_time
        self._status.finished = False
        self._status.current_time = 0.0
        self._status.total_time = 0.0

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
        self._buffs = DataHub(["u"], time_order)
        for _ in range(time_order):
            self._buffs.push_field("u", Sample(0.0, self._fields["u"]))

        # Call callbacks
        for callback in self._callbacks:
            callback.on_task_begin()

    def inference(self) -> SolverStatus:
        start = time.perf_counter()

        # Compute time step
        dt = supports.cfl_timestep(
            self._mesh, self._fields["u"], self._cfl, min_dt=1e-3
        )
        # dt = self._time_step
        rest_time = self._status.end_time - self._status.current_time
        dt = min(dt, self._time_step, rest_time)

        # Call callbacks
        for callback in self._callbacks:
            callback.on_step_begin()

        # Update solution
        old_u = self._fields["u"]
        u_grad, u_diff, u_src = None, None, None
        for op in self._operators:
            if op.get_type() == OperatorType.GRAD:
                u_grad = op.run(self._buffs, dt)
            elif op.get_type() == OperatorType.LAPLACIAN:
                u_diff = op.run(self._buffs, dt)
            elif op.get_type() == OperatorType.SRC:
                u_src = op.run(self._buffs, dt)
        new_u = old_u - dt * old_u * u_grad + dt * u_diff + dt * u_src

        # Update status
        time_cost = time.perf_counter() - start
        self._update_status(time_cost, dt)

        self._fields["u"] = new_u
        self._buffs.push_field("u", Sample(self._status.current_time, new_u))

        # Call callbacks
        for callback in self._callbacks:
            callback.on_step_end()

        if self._status.finished:
            for callback in self._callbacks:
                callback.on_task_end()

        return self._status

    def _update_status(self, time_cost: float, dt: float):
        self._status.current_time += dt
        self._status.step_time = time_cost
        self._status.total_time += time_cost
        self._status.time_step = dt

        if self._status.current_time >= self._status.end_time:
            self._status.finished = True
