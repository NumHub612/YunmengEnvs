# -*- encoding: utf-8 -*-
"""
Copyright (C) 2025, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

The divergence operators for the finite volume method.
"""

from yunmeng.solvers.interfaces import (
    IBoundaryCondition,
    IOperator,
    OperatorType,
    BoundaryType,
)
from yunmeng.numerics.mats.linalgs import LinearEqs
from yunmeng.numerics.grids import Grid
from yunmeng.numerics.fields.fields import Field, Variable
from yunmeng.numerics.fields.datahubs import DataHub
from yunmeng.numerics.algos.topos import MeshTopo
from yunmeng.numerics.algos.geoms import MeshGeom
from yunmeng.numerics.algos.parts import MeshPart


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

    def __init__(self):
        self._mesh: Grid = None
        self._topo: MeshTopo = None
        self._geom: MeshGeom = None
        self._part: MeshPart = None

        self._bcs = None
        self._var = ""

    def prepare(self, fields: list[str], mesh: Grid, bounds: dict):
        if not isinstance(mesh, Grid):
            raise ValueError("FVM div01 operator only supports Grid.")

        self._mesh = mesh
        self._topo = self._mesh.get_topo_assistant()
        self._geom = self._mesh.get_geom_assistant()
        self._part = self._mesh.get_part_assistant()

        self._bcs = bounds
        self._var = fields[0]

    def run(self, sources: DataHub) -> Field | LinearEqs:
        data = sources.field(self._var).data
        variable = data.name
        div_eqs = LinearEqs.zeros(self._part, rhs_type=data.dtype, etype=data.etype)
        # Assemble boundary matrix
        for face in self._topo.boundary_faces:
            bc = self._bcs[face][variable]
            FluxC, FluxF, FluxV = self._handle_boundary(face, bc, data)

            fid = face
            cid = self._topo.face_cells[fid][0]

            div_eqs.matrix[cid, cid] += FluxC
            div_eqs.rhs[cid] -= FluxV

        # Assemble interial matrix
        for face in self._topo.internal_faces:
            Sf = self._geom.face_area[face]
            normal = self._geom.face_normal[face]
            cid1, cid2 = self._topo.face_cells[face]
            u1 = data[cid1]
            u2 = data[cid2]
            u = 0.5 * (u1 + u2)
            mf = self._rho * u * Sf * normal

            if mf.data > 0.0:  # left cell is upstream
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

        FluxC, FluxF, FluxV = (
            Variable.scalar(0.0),
            Variable.scalar(0.0),
            Variable.vector(0.0, 0.0, 0.0),
        )
        # convection part
        mf = self._rho * bc_value * Sb * normal
        FluxC = max(mf.data, 0.0)
        FluxF = -max(-mf.data, 0.0)

        return FluxC, FluxF, FluxV

    def _boundary_2nd(self, fid: int, bcs, field: Field):
        bc_flux = bcs[1]
        cid = self._topo.face_cells[fid][0]
        Sb = self._geom.face_area[fid]
        normal = self._geom.face_normal[fid]

        FluxC, FluxF, FluxV = (
            Variable.scalar(0.0),
            Variable.scalar(0.0),
            Variable.vector(0.0, 0.0, 0.0),
        )

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
