# -*- encoding: utf-8 -*-
"""
Copyright (C) 2025, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

The Laplacian operators for the finite volume method.
"""
from core.solvers.interfaces import (
    IBoundaryCondition,
    IOperator,
    OperatorType,
    BoundaryType,
)
from core.numerics.mats import LinearEqs
from core.numerics.fields import Field, Vector, Scalar, DataHub
from core.numerics.mesh import Grid

import numpy as np


class Lap01(IOperator):
    """
    First order central difference scheme for Laplacian operator.

    scheme:
        - Implicit method.
        - Only support `Grid` mesh.
        - The medium was assumed to be isotropic.
    """

    @classmethod
    def get_type(cls) -> OperatorType:
        return OperatorType.LAPLACIAN

    @classmethod
    def get_name(cls) -> str:
        return "lap01"

    def __init__(self, k: float):
        self._mesh = None
        self._topo = None
        self._geom = None

        self._bcs = None
        self._k = k
        self._var = ""

    def prepare(self, vars: list[str], mesh: Grid, boundaries: dict):
        if not isinstance(mesh, Grid):
            raise ValueError("Fvm Grad01 operator only supports Grid.")

        self._mesh = mesh
        self._topo = self._mesh.get_topo_assistant()
        self._geom = self._mesh.get_geom_assistant()

        self._bcs = boundaries
        self._var = vars[0]

    def run(self, source: DataHub) -> Field | LinearEqs:
        source = source.field(self._var).data
        variable = source.variable
        lap_eqs = LinearEqs.zeros(
            self._mesh.cell_count, rhs_type=source.dtype, variable=variable
        )

        # Aseemble boundary matrix
        for face in self._topo.boundary_faces:
            bc = self._bcs[face][variable]
            FluxC, FluxF, FluxV = self._handle_boundary(face, bc)
            fid = self._mesh.faces[face].id
            cid = self._topo.face_cells[fid][0]

            lap_eqs.matrix[cid, cid] += FluxC
            lap_eqs.rhs[cid] -= FluxV

        # Assemble interial matrix
        for face in self._topo.interior_faces:
            fid = self._mesh.faces[face].id
            Sf = self._geom.face_area[face]
            normal = self._geom.face_normal[face]
            if abs(normal.x) > 1e-10:
                sign = 1 if normal.x > 0 else -1
            else:
                sign = 1 if normal.y > 0 else -1
            Sf = sign * Sf

            cid1, cid2 = self._topo.face_cells[fid]
            dist = self._geom.cell2cell_distance[cid1][cid2]
            FluxC = self._k * Sf / dist
            FluxF = -FluxC

            lap_eqs.matrix[cid1, cid1] += FluxC
            lap_eqs.matrix[cid1, cid2] += FluxF
            lap_eqs.matrix[cid2, cid2] += FluxC
            lap_eqs.matrix[cid2, cid1] += FluxF

        return lap_eqs

    def _handle_boundary(self, fid: int, bc: IBoundaryCondition):
        items = bc.evaluate()
        if bc.get_type() == BoundaryType.FIXED:
            return self._boundary_1st(fid, items)
        elif bc.get_type() == BoundaryType.NATURAL:
            return self._boundary_2nd(fid, items)
        elif bc.get_type() == BoundaryType.MIXED:
            return self._boundary_3rd(fid, items)

    def _boundary_1st(self, fid: int, bcs):
        bc_value = bcs[0]
        cid = self._topo.face_cells[fid][0]
        Sb = self._geom.face_area[fid]
        normal = self._geom.face_normal[fid]
        Ncb = self._geom.cell2face_vector[cid][fid]

        FluxC, FluxF, FluxV = Scalar(), Scalar(), Vector()
        # diffusion part
        sign = 1 if (normal * Ncb).value > 0 else -1
        dist = self._geom.cell2face_distance[cid][fid]
        FluxC += sign * self._k * Sb / dist
        FluxV += -FluxC * bc_value

        return FluxC, FluxF, FluxV

    def _boundary_2nd(self, fid: int, bcs):
        bc_flux = bcs[1]
        Sb = self._geom.face_area[fid]

        FluxC, FluxF, FluxV = Scalar(), Scalar(), Vector()
        # diffusion part
        FluxV = bc_flux * Sb

        return FluxC, FluxF, FluxV

    def _boundary_3rd(self, fid: int, bcs):
        raise NotImplementedError()

        bc_inf, bc_coef, _ = bcs
        cid = self._topo.face_cells[fid][0]
        Sb = self._geom.face_area[fid]
        normal = self._geom.face_normal[fid]

        FluxC, FluxF, FluxV = Scalar(), Scalar(), Vector()
        # diffusion part
        dist = self._geom.cell2face_distance[cid][fid]

        return FluxC, FluxF, FluxV
