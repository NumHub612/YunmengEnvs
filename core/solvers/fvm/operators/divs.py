# -*- encoding: utf-8 -*-
"""
Copyright (C) 2025, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

The divergence operators for the finite volume method.
"""
from core.solvers.interfaces import (
    IBoundaryCondition,
    IOperator,
    OperatorType,
    BoundaryType,
)
from core.numerics.mats import LinearEqs
from core.numerics.fields import Field, DataHub, Vector, Scalar
from core.numerics.mesh import Grid

import numpy as np


class Div01(IOperator):
    """
    First order upwind scheme for divergence operator.

    scheme:
        - Implicit method.
        - Only support `Grid` mesh.
    """

    @classmethod
    def get_type(cls) -> OperatorType:
        return OperatorType.DIV

    @classmethod
    def get_name(cls) -> str:
        return "div01"

    def __init__(self, rho: float):
        self._mesh = None
        self._topo = None
        self._geom = None

        self._bcs = None
        self._rho = rho
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
        source: Field = source.field(self._var).data
        variable = source.variable
        div_eqs = LinearEqs.zeros(
            self._mesh.cell_count, rhs_type=source.dtype, variable=variable
        )
        # Assemble boundary matrix
        for face in self._topo.boundary_faces:
            bc = self._bcs[face][variable]
            FluxC, FluxF, FluxV = self._handle_boundary(face, bc, source)

            fid = self._mesh.faces[face].id
            cid = self._topo.face_cells[fid][0]

            div_eqs.matrix[cid, cid] += FluxC
            div_eqs.rhs[cid] -= FluxV

        # Assemble interial matrix
        for face in self._topo.interior_faces:
            Sf = self._geom.face_area[face]
            normal = self._geom.face_normal[face]
            cid1, cid2 = self._topo.face_cells[face]
            u1 = source[cid1]
            u2 = source[cid2]
            u = 0.5 * (u1 + u2)
            mf = self._rho * u * Sf * normal

            if mf.value > 0.0:  # left cell is upstream
                div_eqs.matrix[cid1, cid1] += mf
                div_eqs.matrix[cid2, cid1] -= mf
            else:  # right cell is upstream
                div_eqs.matrix[cid1, cid2] += mf
                div_eqs.matrix[cid2, cid2] -= mf

        return div_eqs

    def _handle_boundary(self, fid: int, bc: IBoundaryCondition, field: Field):
        items = bc.evaluate()
        if bc.get_type() == BoundaryType.FIXED:
            return self._boundary_1st(fid, items, field)
        elif bc.get_type() == BoundaryType.NATURAL:
            return self._boundary_2nd(fid, items, field)
        elif bc.get_type() == BoundaryType.MIXED:
            return self._boundary_3rd(fid, items, field)

    def _boundary_1st(self, fid: int, bcs, field: Field):
        bc_value = bcs[0]
        Sb = self._geom.face_area[fid]
        normal = self._geom.face_normal[fid]

        FluxC, FluxF, FluxV = Scalar(), Scalar(), Vector()
        # convection part
        mf = self._rho * bc_value * Sb * normal
        FluxC = max(mf.value, 0.0)
        FluxF = -max(-mf.value, 0.0)

        return FluxC, FluxF, FluxV

    def _boundary_2nd(self, fid: int, bcs, field: Field):
        bc_flux = bcs[1]
        cid = self._topo.face_cells[fid][0]
        Sb = self._geom.face_area[fid]
        normal = self._geom.face_normal[fid]

        FluxC, FluxF, FluxV = Scalar(), Scalar(), Vector()

        # convection part
        u = bc_flux
        mf = self._rho * u * Sb * normal

        return FluxC, FluxF, FluxV

    def _boundary_3rd(self, fid: int, bcs, field: Field):
        raise NotImplementedError()

        bc_inf, bc_coef, _ = bcs
        cid = self._topo.face_cells[fid][0]
        Sb = self._geom.face_area[fid]
        normal = self._geom.face_normal[fid]

        FluxC, FluxF, FluxV = Scalar(), Scalar(), Vector()

        # convection part
        mf = self._rho * bc_coef * Sb * normal
        FluxV += mf * bc_inf

        return FluxC, FluxF, FluxV
