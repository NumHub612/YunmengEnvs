# -*- encoding: utf-8 -*-
"""
Copyright (C) 2025, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Solutions for the transient 2D FVM equations.
"""
from core.solvers.commons import BaseSolver, SolverMeta, SolverStatus, SolverType
from core.solvers.commons import inits, boundaries, IBoundaryCondition
from core.numerics.mesh import Grid2D, MeshTopo, MeshGeom
from core.solvers.fvm.operators import Grad01, Grad02
from core.numerics.algos import FieldInterpolators as fis
from core.numerics.fields import Vector, CellField, VariableType
from core.numerics.mats import LinearEqs
from configs.settings import settings, logger

import time
import numpy as np
import copy


class UnsteadyDiffusion(BaseSolver):

    @classmethod
    def get_meta(cls) -> SolverMeta:
        metas = SolverMeta()
        metas.description = "Test solver of the unsteady 2D diffusion equation."
        metas.type = SolverType.FVM
        metas.equation = "Unsteady 2D Diffusion Equation"
        metas.equation_expr = "ddt(rho*phi)-div(k*grad(phi)) == 0"
        metas.dimension = "2d"
        metas.default_ics = {"phi": "uniform(0.0)"}
        metas.default_bcs = {"phi": "constant(0.0, 0.0)"}
        metas.fields = {
            "phi": {
                "description": "scalar field",
                "etype": "cell",
                "dtype": "scalar",
            },
        }
        return metas

    @classmethod
    def get_name(cls) -> str:
        return "unsteady_diffusion"

    def __init__(self, id: str, mesh: Grid2D):
        super().__init__(id, mesh)

        self._geom = mesh.get_geom_assistant()
        self._topo = mesh.get_topo_assistant()

        self._default_bcs = {"phi": boundaries.MixedBoundary("phi", 0.0, 0.0)}
        self._default_ics = {"phi": inits.UniformInitialization("phi", 0.0)}
        self._operators = {"phi": Grad01()}
        self._k = 1.0
        self._rho = 1.0
        self._order = 1
        self._max_iter = 100
        self._tol = 1e-6
        self._step = 0

        self._fields = {
            "phi": CellField(self._mesh.cell_count, VariableType.SCALAR),
            "phi_prev": CellField(self._mesh.cell_count, VariableType.SCALAR),
        }

    def initialize(
        self, k: float, order: int = 1, max_iter: int = 100, tol: float = 1e-6
    ):
        logger.info("Initializing the unsteady 2D diffusion solver...")

        # Check initial conditions
        if "phi" not in self._ics:
            logger.warning(
                f"Solver {self._id} has no initial condition for phi, using default."
            )
            self._ics["phi"] = self._default_ics["phi"]

        # Apply initial conditions
        self._ics["phi"].apply(self._fields["phi"])

        # Check boundary conditions
        for face in self._topo.boundary_faces:
            if face not in self._bcs or "phi" not in self._bcs[face]:
                logger.warning(
                    f"Solver {self._id} has no boundary condition for phi on face \
                     {face}, using default."
                )
                self._bcs[face] = self._default_bcs["phi"]
        self._fields["phi_prev"] = copy.deepcopy(self._fields["phi"])

        # Init or reset status
        self._step = 0
        self._status = SolverStatus()

        # Init parameters
        self._max_iter = max_iter
        self._tol = tol
        self._k = k
        self._order = order

        # Init operators
        for _, op in self._operators.items():
            op.prepare(self._mesh)

        # Call callbacks
        for callback in self._callbacks:
            callback.on_task_begin()

    def inference(self, dt: float = 1.0) -> SolverStatus:
        logger.info("Inference the 2D diffusion solver...")
        start = time.perf_counter()

        sys = LinearEqs.zeros("phi", self._mesh.cell_count)
        # Assemble time matrix
        self._handle_transient(sys, dt, self._order)

        # Aseemble boundary matrix
        for face in self._topo.boundary_faces:
            bc = self._bcs[face]["phi"]
            FluxC, FluxF, FluxV = self._handle_boundary(face, bc)
            fid = self._mesh.faces[face].id
            cid = self._topo.face_cells[fid][0]

            sys.matrix[cid, cid] += FluxC + FluxF
            sys.rhs[cid] -= FluxV

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
            FluxC = self._k * Sf / dist
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
        self._fields["phi_prev"] = copy.deepcopy(self._fields["phi"])
        self._fields["phi"] = solutions
        self._step += 1

        # Update status
        diffs = self._fields["phi"] - self._fields["phi_prev"]
        res = np.max(np.abs(diffs.data))
        self._status.elapsed_time = time.perf_counter() - start
        self._status.finished = res <= self._tol or self._step >= self._max_iter
        self._status.progress = (
            self._step / self._max_iter if not self._status.finished else 1.0
        )
        self._status.converged = True

        # Call callbacks
        for callback in self._callbacks:
            callback.on_step()

        return self.status

    def _handle_transient(self, sys: LinearEqs, dt: float, order: int):
        if order == 1:
            self._handle_transient_1st(sys, dt)
        elif order == 2:
            self._handle_transient_2nd(sys, dt)
        else:
            raise ValueError(f"Unsupported time discretization order: {order}.")

    def _handle_transient_1st(self, sys: LinearEqs, dt: float):
        # FOUE(first-order upwind scheme)
        for cell in self._mesh.cells:
            cid = cell.id
            Vol = self._geom.cell_volumes[cid]
            phi = self._fields["phi"][cid]
            coef = self._rho * Vol / dt

            sys.matrix[cid, cid] += coef
            sys.rhs[cid] += coef * phi

    def _handle_transient_2nd(self, sys: LinearEqs, dt: float):
        # SOUE(second-order upwind scheme)
        for cell in self._mesh.cells:
            cid = cell.id
            Vol = self._geom.cell_volumes[cid]
            phi = self._fields["phi"][cid]
            phi_prev = self._fields["phi_prev"][cid]
            tmp = self._rho * Vol / (2.0 * dt)

            fluxC = 3.0 * tmp
            fluxV = 4.0 * tmp * phi - tmp * phi_prev

            sys.matrix[cid, cid] += fluxC
            sys.rhs[cid] += fluxV

    def _handle_boundary(self, face: int, bc):
        items = bc.evaluate()
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

        FluxC = self._k * sign * Sb / dist
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
        temp = self._k / dist
        Req = Sb * (bc_coef * temp) / (bc_coef + temp)

        FluxC = Req
        FluxF = 0
        FluxV = -Req * bc_inf
        return FluxC, FluxF, FluxV


class UnsteadyConvection(BaseSolver):

    @classmethod
    def get_meta(cls) -> SolverMeta:
        metas = SolverMeta()
        metas.description = "Test solver of unsteady 2D convection equation."
        metas.type = SolverType.FVM
        metas.equation = "Unsteady 2D Convection Equation"
        metas.equation_expr = "ddt(rho*phi)+div(rho*u*phi) == 0"
        metas.dimension = "2d"
        metas.default_ics = {"phi": "UniformInitialization( 0.0)"}
        metas.default_bcs = {"phi": "boundaries.NaturalBoundary(0.0, 0.0)"}
        metas.fields = {
            "phi": {
                "description": "Scalar phi field",
                "etype": "cell",
                "dtype": "scalar",
            }
        }
        return metas

    @classmethod
    def get_name(cls) -> str:
        return "unsteady_convection"

    def __init__(self, id: str, mesh: Grid2D, u: Vector = Vector(1.0, 1.0)):
        super().__init__(id, mesh)

        self._geom = mesh.get_geom_assistant()
        self._topo = mesh.get_topo_assistant()

        self._default_bcs = {
            "phi": boundaries.MixedBoundary("phi", 0.0, Vector()),
        }
        self._default_ics = {
            "phi": inits.UniformInitialization("phi", 0.0),
        }
        self._operators = {"phi": Grad02()}
        self._rho = 1.0
        self._u = u
        self._step = 0
        self._max_iter = 100
        self._tol = 1e-6

        self._fields = {
            "phi": CellField(self._mesh.cell_count, VariableType.SCALAR),
            "phi_prev": CellField(self._mesh.cell_count, VariableType.SCALAR),
        }

    def initialize(self, max_iter: int = 100, tol: float = 1e-6):
        logger.info("Initializing unsteady 2D convection solver...")

        # Check initial conditions
        for field in ["phi"]:
            if field not in self._ics:
                logger.warning(
                    f"FVM Solver {self._id} has no initial condition for {field}, using default."
                )
                self._ics[field] = self._default_ics[field]

        # Apply initial conditions
        self._ics["phi"].apply(self._fields["phi"])

        # Check boundary conditions
        for face in self._topo.boundary_faces:
            if (
                face not in self._bcs or "phi" not in self._bcs[face]
            ):  # TODO: check u either.
                logger.warning(
                    f"FVM Solver {self._id} has no boundary condition for {field} on face \
                     {face}, using default."
                )
                self._bcs[face] = self._default_bcs["phi"]
        self._fields["phi_prev"] = copy.deepcopy(self._fields["phi"])

        # Init or reset status
        self._step = 0
        self._status = SolverStatus()

        # Init parameters
        self._max_iter = max_iter
        self._tol = tol

        # Init operators
        for _, op in self._operators.items():
            op.prepare(self._mesh)

        # Call callbacks
        for callback in self._callbacks:
            callback.on_task_begin()

    def inference(self, dt: float = 1.0) -> SolverStatus:
        logger.info("Inference unsteady 2D convection solver...")
        start = time.perf_counter()

        sys = LinearEqs.zeros("phi", self._mesh.cell_count)
        # Assemble time matrix
        for cell in self._mesh.cells:
            cid = cell.id
            Vol = self._geom.cell_volumes[cid]
            phi = self._fields["phi"][cid]
            coef = self._rho * Vol / dt

            sys.matrix[cid, cid] += coef
            sys.rhs[cid] += coef * phi

        # Assemble boundary matrix
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
            mf = self._rho * self._u * Sf * normal

            cid1, cid2 = self._topo.face_cells[face]
            if mf.value > 0.0:  # left cell is upstream
                sys.matrix[cid1, cid1] += mf
                sys.matrix[cid2, cid1] -= mf
            else:  # right cell is upstream
                sys.matrix[cid1, cid2] += mf
                sys.matrix[cid2, cid2] -= mf

        # Solve linear system
        solutions = sys.solve(method="numpy")

        # Update solution
        self._fields["phi_prev"] = copy.deepcopy(self._fields["phi"])
        self._fields["phi"] = solutions

        # Update status
        diffs = self._fields["phi"] - self._fields["phi_prev"]
        res = np.max(np.abs(diffs.data))
        self._step += 1
        self._status.residual = res
        self._status.elapsed_time = time.perf_counter() - start
        self._status.finished = res <= self._tol or self._step >= self._max_iter
        self._status.progress = (
            self._step / self._max_iter if not self._status.finished else 1.0
        )
        self._status.converged = True

        # Call callbacks
        for callback in self._callbacks:
            callback.on_step()

        return self._status

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
        normal = self._geom.face_normals[face]
        Sf = self._geom.face_areas[face]
        mf = self._rho * self._u * Sf * normal

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
