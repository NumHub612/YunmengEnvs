# -*- encoding: utf-8 -*-
"""
Copyright (C) 2026, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Fdm diffusion operator implementation.
"""

from __future__ import annotations
import numpy as np
import torch


from yunmeng.standard.solver import IOperator, OperatorKinds, OperatorResult
from yunmeng.standard.supports import (
    DataProduct,
    FieldMeta,
    IBackend,
    IDataHub,
    IGrid,
)
from yunmeng.standard.types import ArrayLike, ElementType, MeshDimension
from yunmeng.numerics.fields import Field
from yunmeng.numerics.linalgs import LinearEqs, get_matrix_factory
from yunmeng.numerics.algos import ym_register


@ym_register("operator")
class FdmDiffusionOperator(IOperator):
    """Backward-Euler implicit diffusion operator on 1D structured grid."""

    @classmethod
    def get_name(cls) -> str:
        return "FdmDiffusion"

    @classmethod
    def get_kind(cls) -> str:
        return OperatorKinds.MATH_LAPLACIAN

    def __init__(self, oid: str, field: str, nu: float):
        self._id = oid
        self._field = field
        self._field_loc = ElementType.CELL
        self._nu = float(nu)
        self._n = 0
        self._dx = 0.0
        self._backend = None
        self._differentiable = False
        self._laplacian = None
        self._identity = None

    @property
    def id(self) -> str:
        return self._id

    @property
    def target_fields(self) -> list[str]:
        return [self._field]

    @property
    def time_order(self) -> int:
        return 1

    @property
    def differentiable(self) -> bool:
        return self._differentiable

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

    def build(self, mesh: IGrid, backend: IBackend):
        if not isinstance(mesh, IGrid) or mesh.dimension != MeshDimension.D1:
            raise TypeError(f"Op {self._id!r} mesh must be a 1d IGrid instance")

        self._backend = backend
        self._differentiable = backend.differentiable
        self._n = mesh.element_count(ElementType.CELL)
        self._dx = float(mesh.spacing[0])

        # build laplacian matrix
        lap_mat = np.zeros((self._n, self._n), dtype="float64")
        idx = np.arange(1, self._n - 1)  # 1d grid
        lap_mat[idx, idx] = -2.0 / self._dx**2
        lap_mat[idx, idx - 1] = 1.0 / self._dx**2
        lap_mat[idx, idx + 1] = 1.0 / self._dx**2

        # backend transfer
        mat_cls = get_matrix_factory(backend)
        self._laplacian = mat_cls.from_data(lap_mat, device=backend.device)
        self._identity = mat_cls.identity(self._n, device=backend.device)

    def forward(self, datahub: IDataHub, t: float, dt: float) -> OperatorResult:
        # A=I−Δt*ν*L, b=u^n
        field = datahub.resolve(self.consumes()[0])
        rhs = Field(field.meta, field.values)
        matrix = self._identity - self._laplacian * (dt * self._nu)
        return OperatorResult(implicit=LinearEqs(matrix, rhs))
