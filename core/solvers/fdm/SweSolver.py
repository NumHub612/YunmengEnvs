# -*- encoding: utf-8 -*-
"""
Copyright (C) 2026, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

2D Shallow Water Equations (SWE) solver in FDM format.

The 2D-SWE are the core governing equations in fluid mechanics for describing
free surface flows (such as rivers, lakes, tsunamis, dam-break floods, etc.).
Its core assumption is that:
the vertical scale is much smaller than the horizontal scale
(i.e., the long-wave assumption), so vertical acceleration can be neglected,
and the pressure distribution is approximately hydrostatic.
"""
from core.solvers.commons import (
    BaseSolver,
    SolverMeta,
    SolverStatus,
    SolverType,
    IOperator,
)
from core.solvers.commons.supports.TimeStepAdaptor import cfl_time_step
from core.solvers.commons import inits, boundaries
from core.numerics.mesh.grids import Grid2D
from core.numerics.enums import ElementType, MeshDimension
from core.numerics.fields.fields import Field, VariableType, FieldMeta
from core.numerics.fields.datahubs import DataHub, Sample
from core.numerics.fields.variables import Var
from configs.settings import logger

import time
import numpy as np


class SweSolver(BaseSolver):
    """
    2D shallow water equation solver in fdm format on fixed 2d Grid.
    """

    @classmethod
    def get_meta(cls) -> SolverMeta:
        metas = SolverMeta()
        metas.description = "Fdm solver for 2d shallow water equation"
        metas.type = SolverType.FDM
        metas.equation = "2d shallow water equation in conservation form"
        metas.equation_expr = (
            "ddt(h*U) + div(h*U*U) + grad(0.5*g*sqr(h)) = -g*h*grad(b)"
        )
        metas.dimension = MeshDimension.D2
        metas.default_ics = {
            "U": inits.UniformInitialization,
            "h": inits.UniformInitialization,
        }
        metas.default_bcs = {
            "U": boundaries.WallBoundary,
            "h": boundaries.WallBoundary,
        }
        metas.fields = {
            "U": FieldMeta(
                vtype=VariableType.VECTOR,
                etype=ElementType.NODE,
            ),
            "h": FieldMeta(
                vtype=VariableType.SCALAR,
                etype=ElementType.NODE,
            ),
        }
        return metas

    @classmethod
    def get_name(cls) -> str:
        return "SweFdm2D"

    def __init__(self, id: str, mesh: Grid2D, operators: dict[str, IOperator]):
        super().__init__(id, mesh, operators)
        self._geom = mesh.get_geom_assistant()
        self._topo = mesh.get_topo_assistant()
        self._part = mesh.get_part_assistant()

        self._default_bcs = {
            "U": boundaries.WallBoundary("U"),
            "h": boundaries.WallBoundary("h"),
        }
        self._default_ics = {
            "U": inits.UniformInitialization("U", [0.0, 0.0, 0.0]),
            "h": inits.UniformInitialization("h", 0.0),
        }

        self._time_step = 0.001
        self._cfl = 0.5
        self._buf: DataHub = None
        self._fields = {
            "U": Field(
                self._part.shards,
                VariableType.VECTOR,
                ElementType.NODE,
            ),
            "h": Field(
                self._part.shards,
                VariableType.SCALAR,
                ElementType.NODE,
            ),
        }

    def initialize(self, total_time: float, time_step: float, cfl: float = 0.5):
        available_vars = list(self._fields.keys())

        # Check initial conditions
        for var in available_vars:
            if var not in self._ics:
                logger.warning(
                    f"Solver {self._id} has no initial condition for {var}, using default."
                )
                self._ics[var] = self._default_ics[var]

        # Apply initial conditions
        for var in self._ics.keys():
            self._ics[var].apply(self._fields[var])

        # Check boundary conditions
        for node in self._topo.boundary_nodes:
            for var in available_vars:
                if node not in self._bcs or var not in self._bcs[node]:
                    logger.warning(
                        f"Solver {self._id} has no boundary condition for {var} on node "
                        f"{node}, using default."
                    )
                    self._bcs[node][var] = self._default_bcs[var]

        # Init or reset status
        self._status = SolverStatus()
        self._status.time_step = time_step
        self._status.end_time = total_time
        self._status.finished = False
        self._status.current_time = 0.0
        self._status.total_time = 0.0

        self._cfl = cfl
        self._time_step = time_step

        # Init operators
        for op in self._operators.values():
            op.prepare(available_vars, self._mesh, self._bcs)

        # Init buffer
        time_order = 2
        if "ddt" in self._operators:
            time_order = max(self._operators["ddt"].time_order, 2)
        self._buf = DataHub(available_vars, time_order)
        for _ in range(time_order):
            for var in available_vars:
                self._buf.push_field(var, Sample(0.0, 0.0, self._fields[var]))

        # Call callbacks
        for callback in self._callbacks:
            callback.on_task_begin()

    def inference(self) -> SolverStatus:
        start = time.perf_counter()

        # Call callbacks
        for callback in self._callbacks:
            callback.on_step_begin()

        # Get current fields from buffer
        Nx, Ny = self._mesh.nx, self._mesh.ny
        H = self._fields["h"]
        U = self._fields["U"]
        u, v, _ = U.scalarize()
        u = u.field_shards[0].data.reshape(Nx, Ny)
        v = v.field_shards[0].data.reshape(Nx, Ny)
        h = H.field_shards[0].data.reshape(Nx, Ny)

        # Compute time step
        dt = cfl_time_step(self._mesh, H, U, self._cfl)
        dt = min(dt, self._time_step, self._status.end_time - self._status.current_time)

        # Update fields using Lax-Friedrichs scheme
        new_h, new_u, new_v = self._scheme(h, u, v, dt)
        new_h = new_h.reshape(-1, 1)
        new_u = new_u.flatten()
        new_v = new_v.flatten()

        # Update fields
        self._fields["U"].field_shards[0].data[:, 0] = new_u
        self._fields["U"].field_shards[0].data[:, 1] = new_v
        self._fields["h"].field_shards[0].data = new_h

        # Update buffer
        self._buf.push_field(
            "h", Sample(self._status.current_time + dt, dt, self._fields["h"])
        )
        self._buf.push_field(
            "U", Sample(self._status.current_time + dt, dt, self._fields["U"])
        )

        # Call callbacks
        for callback in self._callbacks:
            callback.on_step_end()

        # Update status
        time_cost = time.perf_counter() - start
        self._update_status(time_cost, dt)
        return self._status

    def _scheme(self, h, u, v, dt):
        g = 9.81
        dx = self._mesh.lx / (self._mesh.nx - 1)
        dy = self._mesh.ly / (self._mesh.ny - 1)
        # conservation variables
        hu = h * u
        hv = h * v

        # fluxes
        # F = [hu, hu^2/h + 0.5gh^2, hu*hv/h] -> 简化为: [hu, hu*u + 0.5gh^2, hu*v]
        F_h = hu
        F_hu = hu * u + 0.5 * g * h**2
        F_hv = hu * v

        G_h = hv
        G_hu = hv * u
        G_hv = hv * v + 0.5 * g * h**2

        # space differences (central difference)
        dF_h_dx = (np.roll(F_h, -1, axis=0) - np.roll(F_h, 1, axis=0)) / (2 * dx)
        dF_hu_dx = (np.roll(F_hu, -1, axis=0) - np.roll(F_hu, 1, axis=0)) / (2 * dx)
        dF_hv_dx = (np.roll(F_hv, -1, axis=0) - np.roll(F_hv, 1, axis=0)) / (2 * dx)

        dG_h_dy = (np.roll(G_h, -1, axis=1) - np.roll(G_h, 1, axis=1)) / (2 * dy)
        dG_hu_dy = (np.roll(G_hu, -1, axis=1) - np.roll(G_hu, 1, axis=1)) / (2 * dy)
        dG_hv_dy = (np.roll(G_hv, -1, axis=1) - np.roll(G_hv, 1, axis=1)) / (2 * dy)

        # neighbor averages (Lax-Friedrichs dissipation)
        # U_new = 0.25 * (U_E + U_W + U_N + U_S) - dt * (dF/dx + dG/dy)
        h_avg = 0.25 * (
            np.roll(h, 1, axis=0)
            + np.roll(h, -1, axis=0)
            + np.roll(h, 1, axis=1)
            + np.roll(h, -1, axis=1)
        )
        hu_avg = 0.25 * (
            np.roll(hu, 1, axis=0)
            + np.roll(hu, -1, axis=0)
            + np.roll(hu, 1, axis=1)
            + np.roll(hu, -1, axis=1)
        )
        hv_avg = 0.25 * (
            np.roll(hv, 1, axis=0)
            + np.roll(hv, -1, axis=0)
            + np.roll(hv, 1, axis=1)
            + np.roll(hv, -1, axis=1)
        )

        # update conservation variables
        h_new = h_avg - dt * (dF_h_dx + dG_h_dy)
        hu_new = hu_avg - dt * (dF_hu_dx + dG_hu_dy)
        hv_new = hv_avg - dt * (dF_hv_dx + dG_hv_dy)

        # boundary handling
        # Dry bed protection: water depth cannot be negative
        h_new = np.maximum(h_new, 1e-6)

        # Restore velocity
        u_new = hu_new / h_new
        v_new = hv_new / h_new

        # Apply boundary conditions
        h_new, u_new, v_new = self._apply_bc(h_new, u_new, v_new)

        return h_new, u_new, v_new

    def _apply_bc(self, h, u, v):
        # 左右边界 (x=0, x=Lx) -> 法向速度反向，切向速度不变，水深镜像
        u[0, :] = -u[1, :]
        u[-1, :] = -u[-2, :]
        v[0, :] = v[1, :]
        v[-1, :] = v[-2, :]
        h[0, :] = h[1, :]
        h[-1, :] = h[-2, :]

        # 上下边界 (y=0, y=Ly)
        v[:, 0] = -v[:, 1]
        v[:, -1] = -v[:, -2]
        u[:, 0] = u[:, 1]
        u[:, -1] = u[:, -2]
        h[:, 0] = h[:, 1]
        h[:, -1] = h[:, -2]
        return h, u, v

    def _update_status(self, time_cost: float, dt: float):
        self._status.current_time += dt
        self._status.step_time = time_cost
        self._status.total_time += time_cost
        self._status.time_step = dt

        if self._status.current_time >= self._status.end_time:
            self._status.finished = True
