# -*- encoding: utf-8 -*-
"""
Copyright (C) 2025, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

2D Burgers equation solver using finite volume method.
"""

from yunmeng.solvers.commons import BaseSolver, SolverMeta, SolverStatus, SolverType
from yunmeng.solvers.commons import inits, boundaries
from yunmeng.numerics.mats.linalgs import LinearEqs
from yunmeng.numerics.fields import Field, Variable
from yunmeng.numerics.fields.datahubs import DataHub, Sample
from yunmeng.numerics.mesh import Mesh
from yunmeng.numerics.enums import VariableType, ElementType, MeshDimension
from yunmeng.setting import logger

import time
import numpy as np


class Burgers2D(BaseSolver):
    """
    2D Burgers equation solver using finite volume method.
    """

    @classmethod
    def get_meta(cls) -> SolverMeta:
        metas = SolverMeta()
        metas.description = "Test solver of the 2d Burgers equation."
        metas.type = SolverType.FVM
        metas.equation = "2d Burgers equation"
        metas.equation_expr = "ddt(rho*u)+div(rho*u*u)-div(k*grad(u))==src(Q(u))"
        metas.dimension = MeshDimension.D2
        metas.default_ics = {"u": "uniform(0.0)"}
        metas.default_bcs = {"u": "neumann(0.0)"}
        metas.fields = {
            "u": {
                "description": "vector field",
                "etype": "cell",
                "dtype": "vector",
            },
        }
        return metas

    @classmethod
    def get_name(cls) -> str:
        return "Burgers2D"

    def __init__(self, id: str, mesh: Mesh, operators: dict):
        """
        Constructor of 2D Burgers equation solver.
        """
        super().__init__(id, mesh, operators)
        self._geom = mesh.get_geom_assistant()
        self._topo = mesh.get_topo_assistant()
        self._part = mesh.get_part_assistant()

        self._default_bcs = {"u": boundaries.MixedBoundary("u", 0.0, 0.0)}
        self._default_ics = {"u": inits.UniformInitialization("u", 0.0)}

        self._engine = None
        self._max_iter = 100
        self._tol = 1e-6
        self._step = 0

        self._buf: DataHub = None
        self._fields = {
            "u": Field(
                self._part,
                VariableType.VECTOR,
                ElementType.CELL,
            ),
        }

    def initialize(self, engine: str = "numpy", max_iter: int = 100, tol: float = 1e-6):
        # Check initial conditions
        if "u" not in self._ics:
            logger.warning(
                f"Solver {self._id} has no initial condition for u, using default."
            )
            self._ics["u"] = self._default_ics["u"]

        # Apply initial conditions
        self._ics["u"].apply(self._fields["u"])

        # Check boundary conditions
        for face in self._topo.boundary_faces:
            if face not in self._bcs or "u" not in self._bcs[face]:
                logger.warning(
                    f"Solver {self._id} has no boundary condition for u on face \
                     {face}, using default."
                )
                if face not in self._bcs:
                    self._bcs[face] = {}
                self._bcs[face]["u"] = self._default_bcs["u"]

        # Init or reset status
        self._step = 0
        self._status = SolverStatus()

        # Init parameters
        # self._engine = get_engine_method(engine)
        self._max_iter = max_iter
        self._tol = tol

        # Init operators
        for _, op in self._operators.items():
            op.prepare(["u"], self._mesh, bounds=self._bcs)

        time_order = max(self._operators["ddt"].time_order, 2)
        self._buf = DataHub(["u"], time_order)
        for _ in range(time_order):
            self._buf.push_field("u", Sample(None, 0.04, self._fields["u"]))

        # Call callbacks
        for callback in self._callbacks:
            callback.on_task_begin()

    def forward(self, dt: float = 1.0) -> SolverStatus:
        start = time.perf_counter()

        # Call callbacks
        for callback in self._callbacks:
            callback.on_step_begin()

        self._buf.push_field("u", Sample(start.real, dt, self._fields["u"]))
        sys = LinearEqs.zeros(self._part, rhs_type=VariableType.VECTOR)

        # Assemble time matrix(ddt)
        # sys_t = self._operators["ddt"].run(self._buf)
        # sys += sys_t

        # Assemble convection matrix(div)
        # sys_c = self._operators["div"].run(self._buf) # wrong
        # sys += sys_c

        # Assemble diffusion matrix(laplacian)
        # sys_d = self._operators["laplacian"].run(self._buf)  # boundary issue
        # sys += sys_d

        # Assemble source term matrix(src)
        # sys_s = self._operators["src"].run(self._buf)
        # sys += sys_s

        # Call callbacks
        for callback in self._callbacks:
            callback.on_step()

        # Solve linear system
        # solutions = sys.solve(self._engine)
        # self._fields["u"] = solutions
        self._step += 1

        # Update status
        # diffs = self._fields["u"] - self._buf.field("u").data
        # res = np.max(np.abs(diffs.data.as_numpy()))
        time_cost = time.perf_counter() - start
        self._update_status(0, time_cost)

        # Call callbacks
        for callback in self._callbacks:
            callback.on_step_end()

        return self._status

    def _update_status(self, res: float, time_cost: float):
        finished = res <= self._tol or self._step >= self._max_iter
        process = self._step / self._max_iter

        self._status.elapsed_time = time_cost
        self._status.finished = finished
        self._status.progress = process
        self._status.converged = True
