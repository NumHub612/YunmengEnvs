# -*- encoding: utf-8 -*-
"""
Copyright (C) 2025, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

2D Burgers equation solver using finite volume method.
"""
from core.solvers.commons import BaseSolver, SolverMeta, SolverStatus, SolverType
from core.solvers.commons import inits, boundaries, IBoundaryCondition
from core.numerics.mesh import Mesh
from core.solvers.fvm.operators import Grad01, Ddt01, Ddt02, Div01, Lap01, Src01
from core.numerics.algos import FieldInterpolators as fis
from core.numerics.fields import Scalar, Vector, CellField, VariableType
from core.numerics.mats import LinearEqs
from configs.settings import settings, logger

import time
import numpy as np
import copy


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
        metas.dimension = "2d"
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

    def __init__(self, id: str, mesh: Mesh):
        """
        Constructor of 2D Burgers equation solver.
        """
        super().__init__(id, mesh)
        self._geom = mesh.get_geom_assistant()
        self._topo = mesh.get_topo_assistant()

        self._default_bcs = {"u": boundaries.MixedBoundary("u", 0.0, 0.0)}
        self._default_ics = {"u": inits.UniformInitialization("u", 0.0)}
        self._operators = {
            "ddt": Ddt01(),
            "grad": Grad01(),
            "div": Div01(),
            "laplacian": Lap01(),
            "src": Src01(),
        }

        self._max_iter = 100
        self._tol = 1e-6
        self._step = 0

        self._fields = {
            "u": CellField(self._mesh.cell_count, VariableType.VECTOR, variable="u"),
        }

    def initialize(
        self, k: float, order: int = 1, max_iter: int = 100, tol: float = 1e-6
    ):
        # TODO: To split SloverParams, OpParams.

        logger.info("Initializing the unsteady burgers solver...")

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
                self._bcs[face] = self._default_bcs["u"]

        # Init or reset status
        self._step = 0
        self._status = SolverStatus()

        # Init parameters
        self._max_iter = max_iter
        self._tol = tol

        # Init operators
        if order != 1:
            self._operators["ddt"] = Ddt02()

        for _, op in self._operators.items():
            op.prepare(self._mesh, boundaries=self._bcs, k=k, rho=1)

        # Call callbacks
        for callback in self._callbacks:
            callback.on_task_begin()

    def inference(self, dt: float = 1.0) -> SolverStatus:
        logger.info("Inference the unsteady burgers solver...")
        start = time.perf_counter()

        # Call callbacks
        for callback in self._callbacks:
            callback.on_step_begin()

        sys = LinearEqs.zeros(
            self._mesh.cell_count, rhs_type=VariableType.VECTOR, variable="u"
        )
        # Assemble time matrix(ddt)
        sys_t = self._operators["ddt"].run(self._fields["u"], dt)
        sys += sys_t

        # Assemble convection matrix(div)
        sys_c = self._operators["div"].run(self._fields["u"])
        sys += sys_c

        # Assemble diffusion matrix(laplacian)
        sys_d = self._operators["laplacian"].run(self._fields["u"])
        sys += sys_d

        # Assemble source term matrix(src)
        sys_s = self._operators["src"].run(self._fields["u"])
        sys += sys_s

        # Call callbacks
        for callback in self._callbacks:
            callback.on_step()

        # Solve linear system
        solutions = sys.solve(method="numpy")

        self._fields["u_prev"] = copy.deepcopy(self._fields["u"])
        self._fields["u"] = solutions
        self._step += 1

        # Update status
        diffs = self._fields["u"] - self._fields["u_prev"]
        res = np.max(np.abs(diffs.data))
        time_cost = time.perf_counter() - start
        self._update_status(res, time_cost)

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
