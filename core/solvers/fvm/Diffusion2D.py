# -*- encoding: utf-8 -*-
"""
Copyright (C) 2025, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Solutions for the 2D diffusion equation using finite volume method.
"""
from core.solvers.commons import BaseSolver, SolverMeta, SolverStatus
from core.solvers.commons import inits, boundaries, callbacks
from core.numerics.mesh import Grid2D, MeshTopo, MeshGeom
from core.numerics.algos import FieldInterpolations as fis
from core.numerics.fields import CellField, VariableType
from configs.settings import settings, logger

import time
import numpy as np
import torch


class Diffusion2D(BaseSolver):

    @classmethod
    def get_meta(cls) -> SolverMeta:
        metas = SolverMeta()
        metas.description = "Test solver of the 2D diffusion equation."
        metas.type = "fvm"
        metas.equation = "Diffusion Equation"
        metas.equation_expr = "-div(k*grad(u)) == src(f)"
        metas.dimension = "2d"
        metas.default_ics = {"u": "uniform(0.0)"}
        metas.default_bcs = {"u": "constant(0.0, 0.0)"}
        metas.fields = {
            "u": {
                "description": "scalar field",
                "etype": "cell",
                "dtype": "scalar",
            },
        }
        return metas

    @classmethod
    def get_name(cls) -> str:
        return "diffusion2d"

    def __init__(self, id: str, mesh: Grid2D):
        super().__init__(id, mesh)

        self._geom = MeshGeom(mesh)
        self._topo = MeshTopo(mesh)

        self._default_bcs = {"u": boundaries.ConstantBoundary("u", 0.0, 0.0)}
        self._default_ics = {"u": inits.UniformInitialization("u", 0.0)}

        self._start_time = time.perf_counter()

        self._fields = {"u": CellField(self._mesh.cell_count, VariableType.SCALAR)}

    def initialize(self):
        logger.info("Initializing the 2D diffusion solver...")

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

        # Init status
        self._status.iteration = 0
        self._status.time_step = None
        self._status.current_time = None
        self._status.elapsed_time = time.perf_counter() - self._start_time
        self._status.convergence = False
        self._status.infos = ""

        # Call callbacks
        for callback in self._callbacks:
            callback.on_task_begin(self.status, self._fields)

    def inference(self) -> tuple[bool, bool, SolverStatus]:
        logger.info("Inference the 2D diffusion solver...")

        # Update boundary conditions

        # Update fields

        # Update status
        self._status.elapsed_time = time.perf_counter() - self._start_time
        self._status.convergence = True

        # Call callbacks
        for callback in self._callbacks:
            callback.on_step(self.status, self._fields)

        return True, False, self.status
