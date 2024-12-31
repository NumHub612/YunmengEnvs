# -*- encoding: utf-8 -*-
"""
Copyright (C) 2024, The YunmengEnvs Contributors. Join us, share your ideas!  

Solving the 3D Burgers equation using finite difference method.
"""
from core.solvers.commons import BaseSolver
from core.numerics.mesh import Mesh, MeshGeom, MeshTopo
from core.numerics.fields import NodeField
from core.numerics.mesh import Grid3D
from configs.settings import logger

import copy
import time


class Burgers3D(BaseSolver):
    """
    The 3D Burgers equation solver.
    """

    @classmethod
    def get_name(cls) -> str:
        return "burgers3d"

    @classmethod
    def get_meta(cls) -> dict:
        metas = super().get_meta()
        metas.update(
            {
                "description": "Test solver of the 3D Burgers equation.",
                "type": "fdm",
                "equation": "Burgers' Equation",
                "equation_expr": "u_t + u*u_x + v*u_y + w*u_z = nu*(u_xx + u_yy + u_zz),\
                    v_t + u*v_x + v*v_y + w*v_z = nu*(v_xx + v_yy + v_zz),\
                    w_t + u*w_x + v*w_y + w*w_z = nu*(w_xx + w_yy + w_zz)",
                "domain": "3D",
                "default_ic": "none",
                "default_bc": "none",
            }
        )
        metas.update(
            {
                "fields": [
                    {
                        "name": "vel",
                        "description": "velocity field",
                        "etype": "node",
                        "dtype": "vector",
                    },
                ],
            }
        )
        return metas

    @property
    def status(self) -> dict:
        return {
            "iteration": 1,
            "elapsed_time": (self._now_time - self._start_time) * 1000,
            "current_time": self._t,
            "time_step": self._dt,
            "convergence": True,
            "error": "",
        }

    def __init__(self, id: str, mesh: Grid3D):
        """
        Initialize the solver.
        """
        super().__init__(id, mesh)

        if not isinstance(mesh, Grid3D):
            raise ValueError("The mesh domain must be 3D grid.")

        self._geom = MeshGeom(mesh)
        self._topo = MeshTopo(mesh)

        self._total_time = 0.0
        self._dt = 0.0
        self._t = 0.0
        self._nu = 0.01
        self._sigma = 0.2
        self._start_time = time.perf_counter()
        self._now_time = self._start_time

        self._dx = None
        self._dy = None
        self._dz = None

    def initialize(self, time_steps: int, nu: float = 0.01, sigma=0.2):
        """
        Initialize the solver.

        Args:
            time_steps: The total time steps of the simulation.
            nu: The viscosity of the Burgers equation.
        """
        self._fields = {"vel": NodeField(self._mesh.node_count, "vector")}

        # Check initial conditions
        if "vel" not in self._ics:
            raise ValueError(f"Solver {self._id} has no initial condition")

        self._ics["vel"].apply(self._fields["vel"])

        # Check boundary conditions
        for node in self._topo.boundary_nodes_indexes:
            if node not in self._bcs or "vel" not in self._bcs[node]:
                raise ValueError(f"Solver {self._id} has no boundary for {node}.")

        # Init configs
        self._nu = nu
        self._sigma = sigma

        min_x, max_x, _ = self._geom.statistics_node_attribute("x")
        min_y, max_y, _ = self._geom.statistics_node_attribute("y")
        min_z, max_z, _ = self._geom.statistics_node_attribute("z")
        self._dx = (max_x - min_x) / self._mesh.nx
        self._dy = (max_y - min_y) / self._mesh.ny
        self._dz = (max_z - min_z) / self._mesh.nz

        self._dt = sigma * self._dx
        self._total_time = time_steps * self._dt
        self._t = 0.0
        self._start_time = time.perf_counter()
        self._now_time = self._start_time

        # Call callbacks
        for callback in self._callbacks:
            callback.on_task_begin()

    def inference(self, dt: float) -> tuple[bool, bool, dict]:
        """
        Inference the solver to get the solutions.

        Args:
            dt: Specified time step.

        Returns:
            A tuple of (is_done, is_terminated, status).
        """
        self._dt = max(dt, 0.0)
        self._now_time = time.perf_counter()

        u = self._fields["vel"]
        new_u = copy.deepcopy(u)

        # Apply boundary conditions
        for node in self._topo.boundary_nodes_indexes:
            for var, bc in self._bcs.get(node, {"vel": self._default_bc}).items():
                if var not in self._fields:
                    continue
                _, val = bc.evaluate(self._t, self._mesh.nodes[node])
                new_u[node] = val

        # Update interior nodes
        for node in self._topo.interior_nodes_indexes:
            nid, sid, eid, wid, tid, bid = self._mesh.retrieve_node_neighborhoods(node)
            p = u[node]
            e, w, n, s, t, b = u[eid], u[wid], u[nid], u[sid], u[tid], u[bid]

            k1x = -p.x * self._dt / self._dx  # non-linear coefficent
            k1y = -p.y * self._dt / self._dy
            k1z = -p.z * self._dt / self._dz
            k2x = self._nu * self._dt / self._dx**2
            k2y = self._nu * self._dt / self._dy**2
            k2z = self._nu * self._dt / self._dz**2

            new_u[node] = (
                p
                + k1x * (p - w)
                + k1y * (p - s)
                + k1z * (p - b)
                + k2x * (e - 2 * p + w)
                + k2y * (n - 2 * p + s)
                + k2z * (t - 2 * p + b)
            )

        # Update solution
        self._fields["vel"] = new_u
        self._t += self._dt
        self._now_time = time.perf_counter()

        # Call callbacks
        for callback in self._callbacks:
            callback.on_step()

        return self._t >= self._total_time, False, self.status
