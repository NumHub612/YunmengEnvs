# -*- encoding: utf-8 -*-
"""
Copyright (C) 2025, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Solutions for the 2D convection equation using finite volume method.
"""
from core.solvers.commons import BaseSolver, SolverMeta, SolverStatus, SolverType
from core.solvers.commons import inits, boundaries
from core.numerics.mesh import Grid2D, MeshTopo, MeshGeom
from core.solvers.fvm.operators import Grad01
from core.numerics.algos import FieldInterpolators as fis
from core.numerics.fields import CellField, VariableType
from core.numerics.matrix import LinearEqs
from configs.settings import settings, logger

import time
import numpy as np


class Convection2D(BaseSolver):

    @classmethod
    def get_meta(cls) -> SolverMeta:
        metas = SolverMeta()
        metas.description = "Test solver of the 2D convection equation."
        metas.type = SolverType.FVM
        metas.equation = "Convection Equation"
        metas.equation_expr = "div(rho*u*phi) == 0"
        metas.dimension = "2d"
        metas.default_ics = {"u": "uniform(0.0)"}
        metas.default_bcs = {"u": "constant(0.0, 0.0)"}
        metas.fields = {
            "u": {
                "description": "velocity field",
                "etype": "cell",
                "dtype": "vector",
            },
        }
        return metas

    @classmethod
    def get_name(cls) -> str:
        return "convection2d"

    def __init__(self, id: str, mesh: Grid2D):
        super().__init__(id, mesh)

        self._geom = mesh.get_geom_assistant()
        self._topo = mesh.get_topo_assistant()

        self._default_bcs = {"u": boundaries.ConstantBoundary("u", 0.0, 0.0)}
        self._default_ics = {"u": inits.UniformInitialization("u", 0.0)}
        self._operators = {"u": Grad01()}
        self._phi = 1.0

        self._fields = {
            "u": CellField(self._mesh.cell_count, VariableType.VECTOR),
        }

    def initialize(self, phi: float):
        logger.info("Initializing the 2D convection solver...")

        # Check initial conditions
        if "u" not in self._ics:
            logger.warning(
                f"FVM Solver {self._id} has no initial condition for u, using default."
            )
            self._ics["u"] = self._default_ics["u"]

        # Apply initial conditions
        self._ics["u"].apply(self._fields["u"])

        # Check boundary conditions
        for face in self._topo.boundary_faces:
            if face not in self._bcs or "u" not in self._bcs[face]:
                logger.warning(
                    f"FVM Solver {self._id} has no boundary condition for u on face \
                     {face}, using default."
                )
                self._bcs[face] = self._default_bcs["u"]

        # Init parameters
        self._phi = phi

        # Init operators
        for _, op in self._operators.items():
            op.prepare(self._mesh)

        # Call callbacks
        for callback in self._callbacks:
            callback.on_task_begin()

    def inference(self) -> tuple[bool, bool, SolverStatus]:
        logger.info("Inference the 2D diffusion solver...")
        start = time.perf_counter()

        # Compute gradients
        GradC = self._operators["u"].run(self._fields["u"])
        self._fields["u_grad_c"] = GradC

        GradF = fis.interp_cell_to_face(GradC, self._mesh)
        self._fields["u_grad_f"] = GradF

        sys = LinearEqs.zeros("u", self._mesh.cell_count)
        # Aseemble boundary matrix
        for face in self._topo.boundary_faces:
            bc = self._bcs[face]["u"]
            FluxC, FluxF, FluxV = self._handle_boundary(face, bc)
            fid = self._mesh.faces[face].id
            cid = self._topo.face_cells[fid][0]
            cindex = self._topo.cell_indices[cid]

            sys.matrix[cindex, cindex] += FluxC + FluxF
            sys.rhs[cindex] -= FluxV

        # Assemble interial matrix
        for face in self._topo.interior_faces:
            fid = self._mesh.faces[face].id
            Sf = self._geom.face_areas[face]
            normal = self._geom.face_normals[face]
            if abs(normal.x) > 1e-10:
                sign = 1 if normal.x > 0 else -1
            else:
                sign = 1 if normal.y > 0 else -1
            Sf = sign * Sf

            cid1, cid2 = self._topo.face_cells[fid]
            dist = self._geom.cell2cell_distances[cid1][cid2]
            FluxC = self._phi * Sf / dist
            FluxF = -FluxC
            FluxV = 0

            sys.matrix[cid1, cid1] += FluxC
            sys.matrix[cid1, cid2] += FluxF
            sys.rhs[cid1] -= FluxV

            sys.matrix[cid2, cid2] += FluxC
            sys.matrix[cid2, cid1] += FluxF
            sys.rhs[cid2] -= FluxV

        # Solve linear system
        solutions = sys.solve(method="numpy")

        # Update solution
        self._fields["u"] = solutions

        # Update status
        self._status.elapsed_time = time.perf_counter() - start
        self._status.progress = 1.0
        self._status.converged = True
        self._status.finished = True

        # Call callbacks
        for callback in self._callbacks:
            callback.on_step()

        return True, False, self.status

    def _handle_boundary(self, face: int, bc):
        items = bc.evaluate()  # TODO: 统一接口参数设置
        if bc.get_type() == boundaries.BoundaryType.FIXED:
            return self._handle_boundary_1st(face, items)
        elif bc.get_type() == boundaries.BoundaryType.NATURAL:
            return self._handle_boundary_2nd(face, items)
        elif bc.get_type() == boundaries.BoundaryType.MIXED:
            return self._handle_boundary_3rd(face, items)

    def _handle_boundary_1st(self, face: int, bcs):
        bc_value = bcs[0]

        Sb = self._geom.face_areas[face]
        fid = self._mesh.faces[face].id
        normal = self._geom.face_normals[face]
        if abs(normal.x) > 1e-10:
            sign = 1 if normal.x > 0 else -1
        else:
            sign = 1 if normal.y > 0 else -1

        cid = self._topo.face_cells[fid][0]
        dist = self._geom.cell2face_distances[cid][fid]

        FluxC = self._phi * sign * Sb / dist
        FluxF = 0
        FluxV = -FluxC * bc_value
        return FluxC, FluxF, FluxV

    def _handle_boundary_2nd(self, face: int, bcs):
        bc_flux = bcs[1]
        normal = self._geom.face_normals[face]
        if abs(normal.x) > 1e-10:
            sign = 1 if normal.x > 0 else -1
        else:
            sign = 1 if normal.y > 0 else -1

        Sb = self._geom.face_areas[face]
        FluxC, FluxF = 0, 0
        FluxV = bc_flux * Sb * sign
        return FluxC, FluxF, FluxV

    def _handle_boundary_3rd(self, face: int, bcs):
        bc_inf, bc_coef, _ = bcs
        Sb = self._geom.face_areas[face]
        normal = self._geom.face_normals[face]
        if abs(normal.x) > 1e-10:
            sign = 1 if normal.x > 0 else -1
        else:
            sign = 1 if normal.y > 0 else -1
        Sb = sign * Sb

        fid = self._mesh.faces[face].id
        cid = self._topo.face_cells[fid][0]
        dist = self._geom.cell2face_distances[cid][fid]
        temp = self._phi / dist
        Req = Sb * (bc_coef * temp) / (bc_coef + temp)

        FluxC = Req
        FluxF = 0
        FluxV = -Req * bc_inf
        return FluxC, FluxF, FluxV
