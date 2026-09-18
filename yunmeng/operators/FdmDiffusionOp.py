# -*- encoding: utf-8 -*-
"""
Copyright (C) 2026, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

description
"""

from __future__ import annotations
import numpy as np
import torch


from yunmeng.interfaces.solver import IOperator, OperatorKinds, OperatorResult
from yunmeng.interfaces.supports import (
    DataProduct,
    FieldMeta,
    IBackend,
    IDataHub,
    IGrid,
    TOPO_NONE,
)
from yunmeng.interfaces.types import ArrayLike, ElementType
from yunmeng.numerics.fields import Field
from yunmeng.numerics.linalgs import LinearEqs

# TODO: automatically get matrix
from yunmeng.numerics.linalgs import NumpyMatrix, TorchMatrix


class FdmDiffusionOperator(IOperator):
    """Backward-Euler diffusion on a uniform 1D structured grid."""

    def __init__(self, field: str, nu: float):
        self._field = field
        self._nu = float(nu)
        self._n = 0
        self._dx = 0.0
        self._backend = None
        self._laplacian = None
        self._identity = None

    @classmethod
    def get_name(cls) -> str:
        return "FdmDiffusion"

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
        return self._backend is not None and self._backend.differentiable

    @classmethod
    def produces(cls, fname: str, loc: ElementType) -> list[DataProduct]:
        return [DataProduct(name=f"laplacian:{fname}", loc=loc)]

    @classmethod
    def consumes(cls, fname: str, loc: ElementType) -> list[DataProduct]:
        return []

    @property
    def target_fields(self) -> list[str]:
        return [self._field]

    @property
    def time_order(self) -> int:
        return 1

    def build(self, mesh: IGrid, backend: IBackend):
        self._backend = backend
        self._n = mesh.element_count(ElementType.CELL)
        self._dx = float(mesh.spacing[0])
        topo = mesh.get_topo_assistant()

        lap = np.zeros((self._n, self._n), dtype="float64")
        for cell in range(self._n):
            neighbors = [
                j for j in topo.neighbors(cell, ElementType.CELL) if j != TOPO_NONE
            ]
            if len(neighbors) < 2:
                continue
            lap[cell, cell] = -2.0 / self._dx**2
            for neighbor in neighbors:
                lap[cell, neighbor] += 1.0 / self._dx**2

        if backend.name == "torch":
            self._laplacian = TorchMatrix.from_data(lap, device=str(backend.device))
            eye = torch.eye(self._n, dtype=torch.float64, device=backend.device)
            self._identity = TorchMatrix.from_data(eye, device=str(backend.device))
        else:
            self._laplacian = NumpyMatrix.from_data(lap)
            self._identity = NumpyMatrix.identity(self._n)

    def forward(self, datahub: IDataHub, t: float, dt: float) -> OperatorResult:
        field = datahub.get_field(self._field)
        matrix = self._identity - self._laplacian * (dt * self._nu)
        rhs = Field(field.meta, field.values)
        return OperatorResult(explicit=None, implicit=LinearEqs(matrix, rhs))
