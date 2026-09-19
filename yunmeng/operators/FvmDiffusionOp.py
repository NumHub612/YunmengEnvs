# -*- encoding: utf-8 -*-
"""
Copyright (C) 2026, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

FvmDiffusionOperator: cell-centered finite-volume diffusion discretization.

Physics (mechanistic) term, implicit part only:
    d/dt ∫_V u dV = ∮_∂V nu * grad(u) · n dA
discretized cell-wise as
    V_c * du_c/dt = sum_f nu * A_f * (u_nb - u_c) / d_f
and advanced with backward Euler: (I - dt*nu*L_fvm) u' = u.

Unlike the FDM operator (stencil-driven, uniform-mesh only), the matrix
here is assembled from face fluxes through the topo/geom assistants, so
the same code path covers non-uniform and (eventually) unstructured
meshes. Dirichlet cells keep identity rows — constrained values are
scatter-written by the solver before the solve.

Differentiable when the backend is (L_fvm is a constant matrix assembled
once per mesh; the solve carries AD through LinearEqs' engine).
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from yunmeng.interfaces.solver import (
    IOperator,
    OperatorKinds,
    OperatorResult,
)
from yunmeng.interfaces.supports import (
    DataProduct,
    FieldMeta,
    IField,
    IBackend,
    IDataHub,
    IMesh,
    TOPO_NONE,
)
from yunmeng.interfaces.types import ArrayLike, ElementType, VariableType
from yunmeng.numerics.fields import Field
from yunmeng.numerics.linalgs import LinearEqs, NumpyMatrix, TorchMatrix


class FvmDiffusionOperator(IOperator):
    """Backward-Euler finite-volume diffusion operator.

    build(): assembles the FVM Laplacian COO structure once per mesh from
    face connectivity (topo) and face areas / cell volumes / centroid
    distances (geom). forward(): scales the frozen structure by dt*nu and
    hands the implicit system to the solver as LinearEqs.
    """

    def __init__(self, field: str, nu: float):
        self._field = field
        self._nu = float(nu)
        self._n = 0
        self._backend: IBackend = None
        # frozen COO structure of L_fvm (values already include the
        # A_f / (V_c * d_f) weights, NOT nu and NOT dt)
        self._rows: np.ndarray = None
        self._cols: np.ndarray = None
        self._vals: np.ndarray = None
        self._diag_pos: np.ndarray = None  # positions of (c, c) entries
        self._interior: np.ndarray = None
        self._device = "cpu"

    # -- metadata -----------------------------------

    @classmethod
    def get_name(cls) -> str:
        return "FvmDiffusion"

    @classmethod
    def get_kind(cls) -> str:
        return OperatorKinds.MATH_LAPLACIAN

    @property
    def explicit_part(self) -> bool:
        return False

    @property
    def implicit_part(self) -> bool:
        return True

    @property
    def differentiable(self) -> bool:
        return True

    @classmethod
    def produces(cls, fname: str, loc: ElementType) -> list[DataProduct]:
        return [DataProduct(name=f"laplacian:{fname}", loc=loc)]

    @classmethod
    def consumes(cls, fname: str, loc: ElementType) -> list[DataProduct]:
        return []

    @property
    def target_fields(self) -> Sequence[str]:
        return (self._field,)

    @property
    def time_order(self) -> int:
        return 1

    # -- lifecycle ----------------------------------

    def build(self, mesh: IMesh, backend: IBackend):
        """Static phase: assemble the FVM Laplacian structure once per mesh.

        Per interior cell c and per face f of c with neighbor nb:
            w_f = A_f / (V_c * d_f)
            L[c, c]  -= w_f
            L[c, nb] += w_f
        Boundary (Dirichlet) cells keep an identity row; the solver
        scatter-writes the constrained value into rhs before the solve.
        """
        self._backend = backend
        self._n = mesh.element_count(ElementType.CELL)
        self._device = getattr(backend, "device", "cpu") or "cpu"
        topo = mesh.get_topo_assistant()
        geom = mesh.get_geom_assistant()
        centers = np.asarray(geom.coordinates(ElementType.CELL), dtype="float64")

        rows, cols, vals = [], [], []
        interior = []
        for c in range(self._n):
            nbs = topo.neighbors(c, ElementType.CELL)
            if any(nb == TOPO_NONE for nb in nbs):
                # boundary cell: identity row, Dirichlet handled by solver
                rows.append(c)
                cols.append(c)
                vals.append(0.0)  # placeholder; identity added in forward
                continue
            interior.append(c)
            vol = float(np.asarray(geom.cell_volume(c)))
            diag = 0.0
            for face in topo.connectivity(c, ElementType.CELL, ElementType.FACE):
                minus, plus = topo.neighbors(face, ElementType.FACE)
                nb = plus if minus == c else minus
                if nb == TOPO_NONE:
                    # domain boundary face of an interior cell: treated as
                    # a zero-gradient (insulated) face — no flux entry.
                    continue
                area = float(np.asarray(geom.face_area(face)))
                dist = float(np.linalg.norm(centers[nb] - centers[c]))
                w = area / (vol * dist)
                diag -= w
                rows.append(c)
                cols.append(int(nb))
                vals.append(w)
            rows.append(c)
            cols.append(c)
            vals.append(diag)

        self._interior = np.asarray(interior, dtype="int64")
        self._rows = np.asarray(rows, dtype="int64")
        self._cols = np.asarray(cols, dtype="int64")
        self._vals = np.asarray(vals, dtype="float64")
        self._diag_pos = np.nonzero(self._rows == self._cols)[0]

    def forward(self, datahub: IDataHub, t: float, dt: float) -> OperatorResult:
        """Runtime: A = I - dt*nu*L_fvm; rhs = u (BC rows already hold the
        scatter-written constrained values, matching the identity rows)."""
        u = datahub.get_field(self._field).values

        values = -dt * self._nu * self._vals
        values[self._diag_pos] += 1.0  # identity everywhere, incl. BC rows
        if self._backend.name == "torch":
            matrix = TorchMatrix.from_coo(
                (self._n, self._n),
                values,
                self._rows,
                self._cols,
                device=str(self._device),
            )
        else:
            matrix = NumpyMatrix.from_coo(
                (self._n, self._n),
                values,
                self._rows,
                self._cols,
            )
        rhs = u.clone() if hasattr(u, "clone") else np.array(u, copy=True)
        meta = FieldMeta(
            name=self._field,
            vtype=VariableType.SCALAR,
            loc=ElementType.CELL,
            btype=self._backend.name,
        )
        return OperatorResult(
            explicit=None,
            implicit=LinearEqs(matrix, Field(meta, rhs)),
        )

    def __repr__(self) -> str:
        return f"FvmDiffusionOperator(field={self._field!r}, nu={self._nu})"
