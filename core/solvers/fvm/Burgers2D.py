# -*- encoding: utf-8 -*-
"""
Copyright (C) 2025, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

2D Burgers equation solver using finite volume method.
"""
from core.solvers.commons import BaseSolver, SolverMeta, SolverStatus, SolverType
from core.solvers.commons import inits, boundaries, IBoundaryCondition
from core.numerics.mesh import Mesh
from core.solvers.fvm.operators import Grad01, Ddt01, Ddt02, Div01, Lap01
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
        }

        self._k = 1.0
        self._rho = 1.0
        self._order = 1

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
        self._k = k
        self._order = order

        # Init operators
        if order != 1:
            self._operators["ddt"] = Ddt02()

        for _, op in self._operators.items():
            op.prepare(self._mesh, boundaries=self._bcs, k=self._k)

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
        sys_t = self._operators["ddt"].run(self._fields["u"], dt, self._rho)
        sys += sys_t

        # Assemble convection matrix(div)
        sys_c = self._handle_convection_term()
        sys += sys_c

        # Assemble diffusion matrix(laplacian)
        sys_d = self._operators["laplacian"].run(self._fields["u"])
        sys += sys_d

        # Assemble source term matrix(src)
        # sys_s = self._handle_source_term()
        # sys += sys_s

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

    def _handle_source_term(self):
        sys = LinearEqs.zeros(
            self._mesh.cell_count, rhs_type=VariableType.VECTOR, variable="u"
        )
        return sys

    def _handle_convection_term(self):
        sys = LinearEqs.zeros(
            self._mesh.cell_count, rhs_type=VariableType.VECTOR, variable="u"
        )
        # Assemble boundary matrix
        for face in self._topo.boundary_faces:
            bc = self._bcs[face]["u"]
            FluxC, FluxF, FluxV = self._handle_boundary_c(face, bc)

            fid = self._mesh.faces[face].id
            cid = self._topo.face_cells[fid][0]

            sys.matrix[cid, cid] += FluxC
            sys.rhs[cid] -= FluxV

        # Assemble interial matrix
        for face in self._topo.interior_faces:
            Sf = self._geom.face_areas[face]
            normal = self._geom.face_normals[face]
            cid1, cid2 = self._topo.face_cells[face]
            u1 = self._fields["u"][cid1]
            u2 = self._fields["u"][cid2]
            u = 0.5 * (u1 + u2)
            mf = self._rho * u * Sf * normal

            if mf.value > 0.0:  # left cell is upstream
                sys.matrix[cid1, cid1] += mf
                sys.matrix[cid2, cid1] -= mf
            else:  # right cell is upstream
                sys.matrix[cid1, cid2] += mf
                sys.matrix[cid2, cid2] -= mf

        return sys

    def _handle_boundary_c(self, fid: int, bc: IBoundaryCondition):
        items = bc.evaluate()
        if bc.get_type() == boundaries.BoundaryType.FIXED:
            return self._boundary_1st_c(fid, items)
        elif bc.get_type() == boundaries.BoundaryType.NATURAL:
            return self._boundary_2nd_c(fid, items)
        elif bc.get_type() == boundaries.BoundaryType.MIXED:
            return self._boundary_3rd_c(fid, items)

    def _boundary_1st_c(self, fid: int, bcs):
        bc_value = bcs[0]
        cid = self._topo.face_cells[fid][0]
        Sb = self._geom.face_areas[fid]
        normal = self._geom.face_normals[fid]

        FluxC, FluxF, FluxV = Scalar(), Scalar(), Vector()

        # convection part
        u = self._fields["u"][cid]  # TODO: interpolate from cell center
        mf = self._rho * u * Sb * normal
        if abs(normal.x) > 1e-10:
            sign = 1 if normal.x > 0 else -1
        else:
            sign = 1 if normal.y > 0 else -1

        FluxV += -mf * bc_value if sign > 0.0 else mf * bc_value
        return FluxC, FluxF, FluxV

    def _boundary_2nd_c(self, fid: int, bcs):
        bc_flux = bcs[1]
        cid = self._topo.face_cells[fid][0]
        Sb = self._geom.face_areas[fid]
        normal = self._geom.face_normals[fid]

        FluxC, FluxF, FluxV = Scalar(), Scalar(), Vector()

        # convection part
        u = bc_flux
        mf = self._rho * u * Sb * normal
        if abs(normal.x) > 1e-10:
            sign = 1 if normal.x > 0 else -1
        else:
            sign = 1 if normal.y > 0 else -1
        FluxC = mf if sign > 0.0 else 0.0

        return FluxC, FluxF, FluxV

    def _boundary_3rd_c(self, fid: int, bcs):
        bc_inf, bc_coef, _ = bcs
        cid = self._topo.face_cells[fid][0]
        Sb = self._geom.face_areas[fid]
        normal = self._geom.face_normals[fid]

        FluxC, FluxF, FluxV = Scalar(), Scalar(), Vector()

        # convection part
        mf = self._rho * bc_coef * Sb * normal
        FluxV += mf * bc_inf

        return FluxC, FluxF, FluxV
