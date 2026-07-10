# -*- encoding: utf-8 -*-
"""
Copyright (C) 2024, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Interface for describing and discretizing pde equations.

The functionality of `Equation` overlaps with that of `Solver`. The former
is used for users customize problems and
provide standardized, configurable numerical discretization schemes, which
are also driven by the `Solver`.
For known problems, more efficient solver can be directly developed.
"""

from yunmeng.numerics.enums import VariableType
from yunmeng.numerics.linalgs import LinearEqs
from yunmeng.numerics.fields import Field
from yunmeng.numerics.mesh import Mesh
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class EqSymbol:
    """
    The symbol used in the equation.
    """

    description: str  # Brief description about the symbol.
    dtype: VariableType  # Data type of the symbol.
    coefficient: bool  # Whether the symbol is coefficient.
    bounds: tuple  # Low-high boundarys of the symbol.


class IEquation(ABC):
    """
    Interface for describing and discretizing pde equation.
    """

    @property
    @abstractmethod
    def id(self) -> str:
        """
        The equation id.
        """
        pass

    @property
    def symbols(self) -> dict[str, EqSymbol]:
        """
        The symbols used in the equation.
        """
        pass

    @abstractmethod
    def set_equations(self, equations: list[str], symbols: dict[str, EqSymbol]):
        """
        Setup the equation and symbols for the PDE.
        """
        pass

    @abstractmethod
    def set_coefficients(self, coefficients: dict):
        """
        Set the coefficients.
        """
        pass

    @abstractmethod
    def set_fields(self, fields: dict[str, Field]):
        """
        Set the variable fields.
        """
        pass

    @abstractmethod
    def set_mesh(self, mesh: Mesh):
        """
        Set the domain mesh.
        """
        pass

    @abstractmethod
    def discretize(self) -> LinearEqs:
        """
        Discretize the equations system.
        """
        pass

    @abstractmethod
    def summary(self) -> str:
        """
        Print a summary of the equation.
        """
        pass
