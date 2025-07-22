# -*- encoding: utf-8 -*-
"""
Copyright (C) 2025, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Solutions for the 2D diffusion equation using finite volume method.
"""
from core.solvers.commons import BaseSolver, SolverMeta, SolverStatus, SolverType
from core.solvers.commons import inits, boundaries
from core.numerics.mesh import Grid2D, MeshTopo, MeshGeom
from core.numerics.algos import FieldInterpolations as fis
from core.numerics.fields import CellField, VariableType
from core.numerics.matrix import LinearEqs
from configs.settings import settings, logger

import time
import numpy as np


class Diffusion2D(BaseSolver):

    @classmethod
    def get_meta(cls) -> SolverMeta:
        metas = SolverMeta()
        metas.description = "Test solver of the 2D diffusion equation."
        metas.type = SolverType.FVM
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

        self._fields = {
            "u": CellField(self._mesh.cell_count, VariableType.SCALAR),
        }

    def initialize(self, phi: float):
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

        # Init parameters
        self._phi = phi

        # Call callbacks
        for callback in self._callbacks:
            callback.on_task_begin(self.status, self._fields)

    def inference(self) -> tuple[bool, bool, SolverStatus]:
        logger.info("Inference the 2D diffusion solver...")

        sys = LinearEqs.zeros("u", self._mesh.cell_count)
        # Aseemble boundary matrix
        for face in self._topo.boundary_faces:  # TODO: iteration by face not cell.
            bc = self._bcs[face]["u"]
            FluxC, FluxF, FluxV = self._handle_boundary(face, bc)  # TODO: uniform args.
            fid = self._mesh.faces[face].id
            cid = self._topo.face_cells[fid][0]
            cindex = self._topo.cell_indices[cid]  # TODO: cid == cindex.

            sys.matrix[cindex, cindex] += FluxC + FluxF
            sys.rhs[cindex] -= FluxV

        # Assemble interial matrix
        for face in self._topo.interior_faces:  # iteration by face.
            fid = self._mesh.faces[face].id
            Sf = self._geom.face_areas[face]
            cid1, cid2 = self._topo.face_cells[fid]
            dist = self._geom.cell2cell_distances[cid1][cid2]
            FluxC = self._phi * Sf / dist
            FluxF = -FluxC
            FluxV = 0

            cindex1 = self._topo.cell_indices[cid1]  # TODO: define the face owner.
            cindex2 = self._topo.cell_indices[cid2]
            sys.matrix[cindex1, cindex1] += FluxC
            sys.matrix[cindex1, cindex2] = FluxF
            sys.rhs[cindex1] -= FluxV

        # Solve linear system
        solutions = sys.solve(method="cupy")

        # Update solution
        self._fields["u"] = solutions

        # Update status
        self._status.elapsed_time = time.perf_counter() - self._start_time
        self._status.convergence = True

        # Call callbacks
        for callback in self._callbacks:
            callback.on_step(self.status, self._fields)

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
        cid = self._topo.face_cells[fid][0]
        dist = self._geom.cell2face_distances[cid][fid]

        FluxC = self._phi * Sb / dist
        FluxF = 0
        FluxV = -FluxC * bc_value
        return FluxC, FluxF, FluxV

    def _handle_boundary_2nd(self, face: int, bcs):
        bc_flux = bcs[1]

        Sb = self._geom.face_areas[face]
        FluxC, FluxF = 0, 0
        FluxV = bc_flux * Sb
        return FluxC, FluxF, FluxV

    def _handle_boundary_3rd(self, face: int, bcs):
        bc_inf, bc_coef, _ = bcs
        Sb = self._geom.face_areas[face]
        fid = self._mesh.faces[face].id
        cid = self._topo.face_cells[fid][0]
        dist = self._geom.cell2face_distances[cid][fid]
        temp = self._phi / dist
        Req = Sb * (bc_coef * temp) / (bc_coef + temp)

        FluxC = Req
        FluxF = 0
        FluxV = -Req * bc_inf
        return FluxC, FluxF, FluxV
