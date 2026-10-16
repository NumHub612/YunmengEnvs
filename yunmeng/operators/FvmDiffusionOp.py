# -*- encoding: utf-8 -*-
"""
Copyright (C) 2026, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

FvmDiffusionOperator: cell-centered finite-volume diffusion discretization.

Physics (mechanistic) term, implicit part only:
    d/dt ∫_V u dV = ∮_∂V nu * grad(u) · n dA
discretized cell-wise as
    V_c * du_c/dt = sum_f nu * A_f * (u_nb - u_c) / d_f
and advanced with backward Euler: (I - dt*nu*L_fvm) u' = u.
"""

from __future__ import annotations

from typing import Sequence
import numpy as np

from yunmeng.standard.solver import (
    IOperator,
    OperatorKinds,
    OperatorResult,
)
from yunmeng.standard.supports import (
    DataProduct,
    FieldMeta,
    IField,
    IBackend,
    IDataHub,
    IMesh,
    IGrid,
    TOPO_NONE,
)
from yunmeng.standard.types import ElementType, VariableType, MeshDimension
from yunmeng.numerics.fields import Field
from yunmeng.numerics.linalgs import LinearEqs, get_matrix_factory


class FvmDiffusionOperator(IOperator):
    """Backward-Euler finite-volume diffusion operator."""

    @classmethod
    def get_name(cls) -> str:
        return "FvmDiffusion"

    @classmethod
    def get_kind(cls) -> str:
        return OperatorKinds.MATH_LAPLACIAN

    def __init__(self, oid: str, field: str, nu: float):
        self._id = oid
        self._field = field
        self._field_loc = ElementType.CELL
        self._nu = float(nu)
        self._n = 0
        self._backend: IBackend = None
        self._differentiable = False

        # frozen COO structure of L_fvm (values already include the
        # A_f / (V_c * d_f) weights, NOT nu and NOT dt)
        self._rows: np.ndarray = None
        self._cols: np.ndarray = None
        self._vals: np.ndarray = None
        self._diag_pos: np.ndarray = None  # positions of (c, c) entries
        self._interior: np.ndarray = None

    @property
    def id(self) -> str:
        return self._id

    @property
    def differentiable(self) -> bool:
        return self._differentiable

    @property
    def target_fields(self) -> Sequence[str]:
        return (self._field,)

    @property
    def time_order(self) -> int:
        return 1

    def consumes(self) -> list[DataProduct]:
        return [
            DataProduct(
                kind=OperatorKinds.MATH_VALUE,
                field=self._field,
                loc=self._field_loc,
                time_order=0,
            )
        ]

    def produces(self) -> list[DataProduct]:
        return []

    def build(self, mesh: IMesh, backend: IBackend):
        """Static phase: assemble the FVM Laplacian structure once per mesh.

        Per interior cell c and per face f of c with neighbor nb:
            w_f = A_f / (V_c * d_f)
            L[c, c]  -= w_f
            L[c, nb] += w_f
        Each row "diagonal = -sum of off-diagonal", so the constant vector
        satisfies L1=0 — constant fields generate no flux, and
        conservation is strictly maintained at the discrete level.

        Boundary (Dirichlet) cells keep an identity row; the solver
        scatter-writes the constrained value into rhs before the solve.
        """
        if not isinstance(mesh, IGrid) or mesh.dimension != MeshDimension.D1:
            raise TypeError(f"Op {self._id!r} mesh must be a 1d IGrid instance")

        self._backend = backend
        self._differentiable = backend.differentiable
        self._n = mesh.element_count(ElementType.CELL)
        topo = mesh.get_topo_assistant()
        geom = mesh.get_geom_assistant()
        centers = np.asarray(geom.coordinates(ElementType.CELL), dtype="float64")

        self._interior = topo.interior_elements(ElementType.CELL)
        boundary = topo.boundary_elements(ElementType.CELL)
        rows, cols, vals = [], [], []

        # assign boundary cells to the identity row
        for c in boundary:
            rows.append(int(c))
            cols.append(int(c))
            vals.append(0.0)

        # interior cells: faces and neighbors are in the same order [-x,+x,...]
        for c in self._interior:
            vol = float(geom.cell_volume(c))
            diag = 0.0
            # none-diagonal entries: sum over all faces of cell c
            for face, nb in zip(
                topo.connectivity(c, ElementType.CELL, ElementType.FACE),
                topo.neighbors(c, ElementType.CELL),
            ):
                w = float(geom.face_area(face)) / (
                    vol * np.linalg.norm(centers[nb] - centers[c])
                )  # A_f / (V_c * d_f)
                diag -= w
                rows.append(int(c))
                cols.append(int(nb))
                vals.append(w)
            # diagonal entry: sum of all none-diagonal entries
            rows.append(int(c))
            cols.append(int(c))
            vals.append(diag)

        self._rows = np.asarray(rows, dtype="int64")
        self._cols = np.asarray(cols, dtype="int64")
        self._vals = np.asarray(vals, dtype="float64")
        self._diag_pos = np.nonzero(self._rows == self._cols)[0]

    def forward(self, datahub: IDataHub, t: float, dt: float) -> OperatorResult:
        """Runtime: A = I - dt*nu*L_fvm; rhs = u (BC rows already hold the
        scatter-written constrained values, matching the identity rows)."""
        u = datahub.field(self._field)
        values = -dt * self._nu * self._vals
        values[self._diag_pos] += 1.0  # identity everywhere, incl. BC rows

        mat_cls = get_matrix_factory(self._backend)
        matrix = mat_cls.from_coo(
            (self._n, self._n),
            values,
            self._rows,
            self._cols,
            device=str(self._backend.device),
        )
        rhs = Field(u.meta, u.values)

        return OperatorResult(implicit=LinearEqs(matrix, rhs))
