# -*- encoding: utf-8 -*-
"""
Copyright (C) 2024, The YunmengEnvs Contributors. Join us, share your ideas!  

Solving the 2D Burgers equation using finite difference method.
"""
from core.solvers.interfaces import BaseSolver
from core.numerics.mesh import Mesh, MeshGeom, MeshTopo
from core.numerics.fields import NodeField, Vector
from core.numerics.mesh import Grid2D
from configs.settings import logger

import copy


class Burgers2D(BaseSolver):
    """
    The 2D Burgers equation solver.
    """

    @classmethod
    def get_name(cls) -> str:
        return "burgers2d"

    @classmethod
    def get_meta(cls) -> dict:
        metas = super().get_meta()
        metas.update(
            {
                "description": "Test solver of the 2D Burgers equation.",
                "type": "fdm",
                "equation": "Burgers' Equation",
                "equation_expr": "u_t + u*u_x + v*u_y = nu*(u_xx + u_yy),\
                    v_t + u*v_x + v*v_y = nu*(v_xx + v_yy)",
                "domain": "2D",
                "default_ic": "none",
                "default_bc": "none",
            }
        )
        metas.update(
            {
                "fields": [
                    {"name": "vel", "etype": "node", "dtype": "vector"},
                ],
            }
        )
        return metas

    @property
    def status(self) -> dict:
        return {
            "curr_time": self._t,
            "dt": self._dt,
            "end_time": self._total_time,
        }

    def __init__(self, id: str, mesh: Mesh):
        """
        Initialize the solver.
        """
        super().__init__(id, mesh)

        if not isinstance(mesh, Grid2D):
            raise ValueError("The mesh domain must be 2D grid.")

        self._geom = MeshGeom(mesh)
        self._topo = MeshTopo(mesh)

        self._total_time = 0.0
        self._dt = 0.0
        self._t = 0.0
        self._nu = 0.01
        self._sigma = 0.2

        self._dx = None
        self._dy = None

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
        self._dx = (max_x - min_x) / self._mesh.nx
        self._dy = (max_y - min_y) / self._mesh.ny

        self._dt = sigma * self._dx
        self._total_time = time_steps * self._dt
        self._t = 0.0

        # Call callbacks
        for callback in self._callbacks:
            callback.on_task_begin(self._fields, self._t)

    def inference(self, dt: float) -> tuple[bool, bool, dict]:
        """
        Inference the solver to get the solutions.

        Args:
            dt: Specified time step.

        Returns:
            A tuple of (is_done, is_terminated, status).
        """
        self._dt = max(dt, 0.0)
        self._t += self._dt

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
            nid, sid, eid, wid, _, _ = self._mesh.retrieve_node_neighborhoods(node)
            # eid, wid, nid, sid = self._calc_grid_indexes(node)
            p = u[node]
            e, w, n, s = u[eid], u[wid], u[nid], u[sid]

            k1x = -p.x * self._dt / self._dx  # non-linear coefficent
            k1y = -p.y * self._dt / self._dy
            k2x = self._nu * self._dt / self._dx**2
            k2y = self._nu * self._dt / self._dy**2

            new_u[node] = (
                p
                + k1x * (p - w)
                + k1y * (p - s)
                + k2x * (e - 2 * p + w)
                + k2y * (n - 2 * p + s)
            )

        # Update solution
        self._fields["vel"] = new_u

        # Call callbacks
        for callback in self._callbacks:
            callback.on_step(self._fields, self._t)

        return self._t >= self._total_time, False, self.status

    def _calc_grid_indexes(self, nid: int):
        i = nid // self._mesh.ny
        j = nid % self._mesh.ny

        e_node = (i + 1) * self._mesh.ny + j
        e_node = min(self._mesh.node_count - 1, e_node)

        w_node = (i - 1) * self._mesh.ny + j
        w_node = max(0, w_node)

        n_node = i * self._mesh.ny + (j + 1)
        n_node = min(self._mesh.node_count - 1, n_node)

        s_node = i * self._mesh.ny + (j - 1)
        s_node = max(0, s_node)

        return e_node, w_node, n_node, s_node