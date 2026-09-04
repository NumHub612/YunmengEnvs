# -*- encoding: utf-8 -*-
"""
Copyright (C) 2026, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Interface for describing and discretizing pde equations.

The functionality of `Equation` overlaps with that of `Solver`. The former
is used for users customize problems and
provide standardized, configurable numerical discretization schemes, which
are also driven by the `Solver`.
For known problems, more efficient solver can be directly developed.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from yunmeng.interfaces.support.field import IField
from yunmeng.interfaces.support.mesh import IMesh
from yunmeng.interfaces.types import VariableType
from yunmeng.interfaces.support.linalg import ILinearEqs


@dataclass
class EqSymbol:
    """Symbol used in a configurable equation."""

    description: str  # Brief description about the symbol.
    dtype: VariableType  # Data type of the symbol.
    coefficient: bool  # Whether the symbol is coefficient.
    bounds: tuple  # Low-high boundarys of the symbol.


class IEquation(ABC):
    """Configurable PDE description/discretization (mesh/field types
    protocolized; otherwise unchanged)."""

    @property
    @abstractmethod
    def id(self) -> str: ...

    @property
    def symbols(self) -> dict[str, EqSymbol]: ...

    @abstractmethod
    def set_equations(self, equations: list[str], symbols: dict[str, EqSymbol]): ...

    @abstractmethod
    def set_coefficients(self, coefficients: dict): ...

    @abstractmethod
    def set_fields(self, fields: dict[str, IField]): ...

    @abstractmethod
    def set_mesh(self, mesh: IMesh): ...

    @abstractmethod
    def discretize(self) -> "ILinearEqs": ...

    @abstractmethod
    def summary(self) -> str: ...
