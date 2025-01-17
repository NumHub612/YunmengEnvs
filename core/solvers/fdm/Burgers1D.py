# -*- encoding: utf-8 -*-
"""
Copyright (C) 2024, The YunmengEnvs Contributors. Join us, share your ideas!

Solving the 1D Burgers equation using finite difference method.
"""
from core.solvers.commons import BaseSolver
from core.solvers.commons import inits
from core.numerics.mesh import Mesh, MeshGeom, MeshTopo
from core.numerics.fields import NodeField, Scalar

import copy
import time


class Burgers1D(BaseSolver):
    """
    Solver of the 1D Burgers equation using finite difference method.
    """

    @classmethod
    def get_name(cls) -> str:
        return "Burgers1D"

    @classmethod
    def get_meta(cls) -> dict:
        metas = super().get_meta()
        metas.update(
            {
                "description": "Test solver of the 1D Burgers equation using finite difference method.",
                "type": "fdm",
                "equation": "1D Burgers' Equation",
                "equation_expr": "u_t + u*u_x = nu*u_xx",
                "domain": "1D",
                "default_ic": "uniform",
                "default_bc": "none",
            }
        )
        metas.update(
            {
                "fields": {
                    "u": {
                        "description": "Velocity field",
                        "etype": "node",
                        "dtype": "scalar",
                    }
                },
            }
        )
        return metas

    def __init__(self, id: str, mesh: Mesh):
        """
        Constructor of the Burgers1D solver.
        """
        super().__init__(id, mesh)
        if mesh.dimension != "1d":
            raise ValueError("The domain of the mesh must be 1D.")

        self._geom = MeshGeom(mesh)
        self._topo = MeshTopo(mesh)

        self._default_ic = inits.UniformInitialization("default", Scalar(0.0))
        self._fields = {"u": NodeField(mesh.node_count, "scalar")}

        self._total_time = 0.0
        self._dt = 0.0
        self._t = 0.0
        self._nu = 0.07

        self._start_time = time.perf_counter()
        self._now_time = self._start_time

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

    def initialize(self, time_steps: int, nu: float = 0.07):
        """
        Initialize the solver.

        Args:
            time_steps: The total time steps of the simulation.
            nu: The viscosity of the Burgers equation.
        """
        # initialize fields
        for var, ic in self._ics.items():
            ic.apply(self._fields[var])

        # initialize parameters
        self._nu = nu

        min_dx, _, _ = self._geom.statistics_cell_attribute("volume")
        self._dt = nu * min_dx
        self._total_time = time_steps * self._dt
        self._t = 0.0
        self._start_time = time.perf_counter()
        self._now_time = self._start_time

        # run callbacks
        for callback in self._callbacks:
            callback.on_task_begin(self.status, self._fields)

    def inference(self, dt: float) -> tuple[bool, bool, dict]:
        """
        Inference the solver to get the solution.

        Args:
            dt: Specified time step used for updating to next step.

        Returns:
            Tuple of results with (is_done, is_terminated, status).
        """
        self._dt = max(dt, 0.0)
        self._now_time = time.perf_counter()

        u = self._fields["u"]
        new_u = copy.deepcopy(u)

        # Apply boundary conditions
        for node in self._topo.boundary_nodes_indexes:
            for var, bc in self._bcs.get(node, {"u": self._default_bcs}).items():
                if var not in self._fields:
                    continue
                _, val = bc.evaluate(self._t, self._mesh.nodes[node])
                new_u[node] = val

        # Update interior nodes
        for node in self._topo.interior_nodes_indexes:
            lnode, rnode = self._topo.collect_node_neighbours(node)
            if lnode > rnode:
                lnode, rnode = rnode, lnode

            ldist = self._geom.calucate_node_to_node_distance(node, lnode)
            rdist = self._geom.calucate_node_to_node_distance(node, rnode)
            new_u[node] = (
                u[node]
                - u[node] * self._dt / ldist * (u[node] - u[lnode])
                + self._nu
                * self._dt
                / rdist**2.0
                * (u[rnode] - 2.0 * u[node] + u[lnode])
            )

        # Update solution
        self._fields["u"] = new_u
        self._t += self._dt
        self._now_time = time.perf_counter()

        # Call callbacks
        for callback in self._callbacks:
            callback.on_step(self.status, self._fields)

        return self._t >= self._total_time, False, self.status
