# -*- encoding: utf-8 -*-
"""
Copyright (C) 2026, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Interface for describing and discretizing PDE equations.

Relationship between IEquation, IOperator and ISolver:

- Most solvers are purpose-built for a FIXED, known equation — their
  operator assembly is hard-coded or even fixed entirely (a
  Saint-Venant solver knows its terms).

- IEquation exists for CONFIGURABLE solvers: setups where the equation
  itself is edited at run time through external configuration (e.g.
  experimental teaching: add/remove a diffusion or source term and
  re-run). The solver calls set_problems([...]) before initialize();
  each IEquation then discretizes into linear systems and/or assembles
  the operators the solver will step with.

The boundary is whole-assembly, never per-term mixing: configurable
equations REPLACE the solver's assembly as a unit; a solver must never
silently merge IEquation-produced terms with built-in operators.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from yunmeng.interfaces.supports import IField, IMesh, ILinearEqs
from yunmeng.interfaces.types import VariableType


@dataclass
class EqSymbol:
    """Symbol used in a configurable equation."""

    description: str  # Brief description about the symbol.
    vtype: VariableType  # Data type of the symbol.
    coefficient: bool  # Whether the symbol is coefficient.
    bounds: tuple  # Low-high boundarys of the symbol.


class IEquation(ABC):
    """Configurable PDE description/discretization (mesh/field types
    protocolized; otherwise unchanged)."""

    @property
    @abstractmethod
    def id(self) -> str: ...

    @property
    @abstractmethod
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
