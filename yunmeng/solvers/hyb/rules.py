# -*- encoding: utf-8 -*-
"""
Copyright (C) 2026, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Concrete BC / IC rules for the hybrid-validation demo.
"""

from __future__ import annotations

from typing import Callable
from dataclasses import dataclass
import numpy as np

from yunmeng.interfaces.solver.IBoundaryCondition import IBoundaryCondition
from yunmeng.interfaces.solver.IInitCondition import IInitialCondition
from yunmeng.interfaces.supports.field import IField, Variable
from yunmeng.interfaces.supports.mesh import IRegion
from yunmeng.interfaces.types import ArrayLike, VariableType

from enum import Enum


class VariableType(Enum):
    """Variable tensor rank descriptor."""

    SCALAR = ()
    VECTOR = (3,)
    TENSOR = (3, 3)

    @property
    def shape(self) -> tuple[int, ...]:
        return self.value

    @property
    def ndim(self) -> int:
        return len(self.value)

    @property
    def n_components(self) -> int:
        n = 1
        for d in self.value:
            n *= d
        return n


@dataclass
class Variable:
    """A named, typed value (used by initial conditions)."""

    name: str
    vtype: VariableType = VariableType.SCALAR
    values: ArrayLike = None
    unit: str = ""


class DirichletBC(IBoundaryCondition):
    """Dirichlet rule: constant or time-dependent prescribed value."""

    def __init__(
        self,
        bc_id: str,
        target_field: str,
        region: IRegion,
        value: float | Callable[[float], float],
        tag: str = "open",
    ):
        self._id = bc_id
        self._field = target_field
        self._region = region
        self._value = value
        self._tag = tag

    @classmethod
    def get_name(cls) -> str:
        return "DirichletBC"

    @property
    def semantic_tag(self) -> str:
        return self._tag

    @property
    def target_field(self) -> str:
        return self._field

    @property
    def region(self) -> IRegion:
        return self._region

    @property
    def id(self) -> str:
        return self._id

    def evaluate(self, t: float) -> tuple[str, ArrayLike]:
        v = self._value(t) if callable(self._value) else self._value
        return "constraints", np.full(len(self._region.element_ids), float(v))


class GaussianIC(IInitialCondition):
    """Gaussian initial pulse on a uniform 1D mesh."""

    def __init__(
        self,
        ic_id: str,
        target_field: str,
        centers: ArrayLike,
        center: float = 0.5,
        width: float = 0.08,
        amplitude: float = 1.0,
    ):
        self._id = ic_id
        self._field = target_field
        self._centers = np.asarray(centers, dtype="float64")
        self._c = center
        self._w = width
        self._a = amplitude

    @classmethod
    def get_name(cls) -> str:
        return "GaussianIC"

    @property
    def target_field(self) -> str:
        return self._field

    @property
    def id(self) -> str:
        return self._id

    def get(self, **kwargs) -> Variable:
        return Variable(
            name=self._field,
            vtype=VariableType.SCALAR,
            values=self._a
            * np.exp(-((self._centers - self._c) ** 2) / (2 * self._w**2)),
        )

    def apply(self, field: IField):
        field.values = self.get().values
