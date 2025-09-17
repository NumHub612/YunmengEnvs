# -*- encoding: utf-8 -*-
"""
Copyright (C) 2025, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Solutions for the 2D convection equation using finite volume method.
"""
from core.solvers.commons import BaseSolver, SolverMeta, SolverStatus, SolverType
from core.solvers.commons import inits, boundaries, IBoundaryCondition
from core.numerics.mesh import Grid2D
from core.solvers.fvm.operators import Grad02
from core.numerics.algos import FieldInterpolators as fis
from core.numerics.fields import CellField, VariableType, Vector
from core.numerics.mats import LinearEqs
from core.viewer.plotter import MatPlotters
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
        metas.default_ics = {"phi": "UniformInitialization( 0.0)"}
        metas.default_bcs = {"phi": "boundaries.NaturalBoundary(0.0, 0.0)"}
        metas.fields = {
            "phi": {
                "description": "Scalar phi field",
                "etype": "cell",
                "dtype": "scalar",
            },
            "u": {
                "description": "Vector u field",
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

        self._default_bcs = {
            "phi": boundaries.MixedBoundary("phi", 0.0, Vector()),
            "u": boundaries.MixedBoundary("u", Vector(), Vector()),
        }
        self._default_ics = {
            "phi": inits.UniformInitialization("phi", 0.0),
            "u": inits.UniformInitialization("u", Vector()),
        }
        self._operators = {"phi": Grad02()}
        self._rho = 1.0

        self._fields = {
            "phi": CellField(self._mesh.cell_count, VariableType.SCALAR),
            "u": CellField(self._mesh.cell_count, VariableType.VECTOR),
        }

    def initialize(self):
        logger.info("Initializing the 2D convection solver...")

        # Check initial conditions
        for field in ["phi", "u"]:
            if field not in self._ics:
                logger.warning(
                    f"FVM Solver {self._id} has no initial condition for u, using default."
                )
                self._ics[field] = self._default_ics[field]

        # Apply initial conditions
        self._ics["phi"].apply(self._fields["phi"])
        self._ics["u"].apply(self._fields["u"])

        # Check boundary conditions
        for face in self._topo.boundary_faces:
            if (
                face not in self._bcs or "phi" not in self._bcs[face]
            ):  # TODO: check u either.
                logger.warning(
                    f"FVM Solver {self._id} has no boundary condition for u on face \
                     {face}, using default."
                )
                self._bcs[face] = self._default_bcs["phi"]

        # Init operators
        for _, op in self._operators.items():
            op.prepare(self._mesh)

        # Call callbacks
        for callback in self._callbacks:
            callback.on_task_begin()

    def inference(self) -> tuple[bool, bool, SolverStatus]:
        logger.info("Inference the 2D convection solver...")
        start = time.perf_counter()

        # Interp face u field.
        face_u = fis.interp_cell_to_face(self._fields["u"], self._mesh)
        self._fields["face_u"] = face_u

        sys = LinearEqs.zeros("phi", self._mesh.cell_count)

        # Aseemble boundary matrix
        for face in self._topo.boundary_faces:
            bc = self._bcs[face]["phi"]
            FluxC, FluxF, FluxV = self._handle_boundary(face, bc)

            fid = self._mesh.faces[face].id
            cid = self._topo.face_cells[fid][0]

            sys.matrix[cid, cid] += FluxC
            sys.rhs[cid] -= FluxV

        # Assemble interial matrix
        for face in self._topo.interior_faces:
            Sf = self._geom.face_areas[face]
            normal = self._geom.face_normals[face]
            mf = self._rho * face_u[face] * Sf * normal

            cid1, cid2 = self._topo.face_cells[face]
            if mf.value > 0.0:  # left cell is upstream
                sys.matrix[cid1, cid1] += mf
                sys.matrix[cid2, cid1] -= mf
            else:  # right cell is upstream
                sys.matrix[cid1, cid2] += mf
                sys.matrix[cid2, cid2] -= mf

        # MatPlotters.show_lineareqs_heatmap(
        #     sys,
        #     title="Linear Equation",
        #     cmap="viridis",
        #     figsize=(10, 6),
        #     show=False,
        #     save_dir="./",
        # )

        # Solve linear system
        solutions = sys.solve(method="numpy")

        # Update solution
        self._fields["phi"] = solutions

        # Update status
        self._status.elapsed_time = time.perf_counter() - start
        self._status.progress = 1.0
        self._status.converged = True
        self._status.finished = True

        # Call callbacks
        for callback in self._callbacks:
            callback.on_step()

        return True, False, self.status

    def _handle_boundary(self, face: int, bc: IBoundaryCondition):
        """Boundary for convection problem."""
        items = bc.evaluate()
        if bc.get_type() == boundaries.BoundaryType.FIXED:
            return self._handle_boundary_1st(face, items)
        elif bc.get_type() == boundaries.BoundaryType.NATURAL:
            return self._handle_boundary_2nd(face, items)
        elif bc.get_type() == boundaries.BoundaryType.MIXED:
            return self._handle_boundary_3rd(face, items)

    def _handle_boundary_1st(self, face: int, bcs):
        phi_f = bcs[0]
        phi_u = Vector(1, 1)
        Sf = self._geom.face_areas[face]
        normal = self._geom.face_normals[face]
        mf = self._rho * phi_u * Sf * normal

        FluxC = FluxF = 0.0
        FluxV = -mf * phi_f if mf.value > 0.0 else mf * phi_f
        return FluxC, FluxF, FluxV

    def _handle_boundary_2nd(self, face: int, bcs):
        flux_f = bcs[1]
        cid = self._topo.face_cells[face][0]
        dist = self._geom.cell2face_distances[cid][face]
        phi_c = self._fields["phi"][cid]
        # phi_f = flux_f * dist + phi_c

        phi_u = Vector(1, 1)
        normal = self._geom.face_normals[face]
        Sf = self._geom.face_areas[face]
        mf = self._rho * phi_u * Sf * normal

        FluxC = mf
        FluxF = 0.0
        FluxV = 0.0
        return FluxC, FluxF, FluxV

    def _handle_boundary_3rd(self, face: int, bcs):
        phi_f, v, _ = bcs

        Sf = self._geom.face_areas[face]
        normal = self._geom.face_normals[face]
        mf = self._rho * v * Sf * normal

        FluxC = FluxF = 0.0
        FluxV = mf * phi_f
        return FluxC, FluxF, FluxV
