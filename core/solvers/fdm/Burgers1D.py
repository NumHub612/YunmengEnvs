# -*- encoding: utf-8 -*-
"""
Copyright (C) 2024, The YunmengEnvs Contributors. Join us, for you talents!

Solving the 1D Burgers equation using finite difference method.
"""
from core.solvers.interfaces import ISolver, IInitCondition, IBoundaryCondition
from core.solvers.extensions.inits import UniformInitialization
from core.numerics.mesh import Mesh, MeshGeom, MeshTopo
from core.numerics.fields import Field, NodeField, Scalar

import copy


class Burgers1D(ISolver):
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
                "author": "朗月;",
                "version": "1.0",
                "email": "none;",
                "description": "Solver of the 1D Burgers equation using finite difference method.",
                "type": "fdm",
                "equation": "Burgers' Equation",
                "equation_expr": "u_t + u*u_x = nu*u_xx",
                "domain": "1D",
                "default_init_method": "zero",
                "default_boundary_condition": "wall",
            }
        )
        metas.update(
            {
                "accessibale_fields": [
                    {"name": "u", "etype": "node", "dtype": "scalar"}
                ],
            }
        )
        return metas

    def __init__(self, id: str, mesh: Mesh, *, callbacks: list = None):
        """
        Constructor of the Burgers1D solver.

        Args:
            id: The id of the solver.
            mesh: The mesh of the problem.
            callbacks: The callbacks to be called during the solving process.
        """
        if mesh.domain != "1D":
            raise ValueError("The domain of the mesh must be 1D.")

        self._id = id
        self._mesh = mesh
        self._geom = MeshGeom(mesh)
        self._topo = MeshTopo(mesh)

        self._total_time = 0.0
        self._dt = 0.0
        self._t = 0.0
        self._nu = 0.07
        self._callbacks = callbacks or []

        self._default_init = UniformInitialization("default", Scalar(0.0))
        self._default_bc = None

        self._fields = {"u": NodeField(mesh.node_count, Scalar())}
        self._ics = {}
        self._bcs = {}

    @property
    def id(self) -> str:
        return self._id

    @property
    def current_time(self) -> float:
        return self._t

    @property
    def total_time(self) -> float:
        return self._total_time

    def get_solution(self, field_name: str) -> Field:
        if field_name not in self._fields:
            raise ValueError(f"Invalid field name: {field_name}")

        return self._fields[field_name]

    def set_ic(self, var: str, ic: IInitCondition):
        self._ics[var] = ic

    def set_bc(self, var: str, elems: list, bc: IBoundaryCondition):
        if any(elem.type != "node" for elem in elems):
            raise ValueError("The boundary condition can only be applied to nodes.")

        for node in elems:
            if node not in self._bcs:
                self._bcs[node] = {}
            self._bcs[node][var] = bc

    def initialize(self, total_time: float, nu: float = 0.07):
        """
        Initialize the solver.

        Args:
            total_time: The total time of the simulation, in seconds.
            nu: The viscosity of the Burgers equation.
        """
        for var, ic in self._ics.items():
            ic.apply(self._fields[var])

        self._total_time = total_time
        self._t = 0.0
        self._nu = nu

        min_dx, _, _ = self._mesh.stat_face_area
        self._dt = nu * min_dx

    def update(self):
        u = self._fields["u"]
        new_u = copy.deepcopy(u)

        # Apply boundary conditions
        for node in self._topo.boundary_nodes_indexes:
            for var, bc in self._bcs.get(node, {"u": self._default_bc}).items():
                flux, val = bc.evaluate(self._t, self._mesh.nodes[node])
                new_u[node] = val

        # Update interior nodes
        for node in self._topo.interior_nodes_indexes:
            lnode, rnode = self._topo.collect_cell_neighbours(node)
            ldist = self._geom.calucate_node_to_node_distance(node, lnode)
            rdist = self._geom.calucate_node_to_node_distance(node, rnode)
            new_u[node] = (
                u[node]
                - u[node] * self._dt / ldist * (u[node] - u[lnode])
                + self._nu * self._dt / rdist**2 * (u[rnode] - 2 * u[node] + u[lnode])
            )

        # Update solution
        self._fields["u"] = new_u
        self._t += self._dt

        # Call callbacks
        for callback in self._callbacks:
            status = {"time": self._t, "dt": self._dt}
            results = self._fields
            callback(self, status, results)
