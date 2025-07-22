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
from core.numerics.matrix import LinearEqs
from core.numerics.fields import Field
from core.numerics.mesh import Mesh
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class EqSymbol:
    """
    The symbol used in the equation.
    """

    description: str  # A brief description about the symbol.
    type: str  # Type of the symbol, e.g. "scalar", "vector", "tensor".
    coefficient: bool  # Whether the symbol is coefficient.
    boundary: tuple  # Boundarys of the symbol.


class IEquation(ABC):
    """
    Interface for describing and discretizing pde equations.
    """

    SUPPORTED_OPS = [
        "LAPLACIAN",
        "GRAD",
        "DIV",
        "CURL",
        "DDT",
        "D2DT2",
        "SRC",
        "FUNC",
    ]

    @property
    @abstractmethod
    def id(self) -> str:
        """
        The equation id.
        """
        pass

    @abstractmethod
    def set_equations(self, equations: list[str], symbols: dict[str, EqSymbol]):
        """
        Set the equations and symbols for the equation.
        """
        pass

    @abstractmethod
    def set_coefficients(self, coefficients: dict):
        """
        Set the coefficients for the equations.
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
    def get_variables(self) -> dict:
        """
        Get the variables used in equations.
        """
        pass

    @abstractmethod
    def discretize(self) -> LinearEqs:
        """
        Discretize the equations system.
        """
        pass


class IOperator(ABC):
    """
    Interface for discretizing pde term to a numerical form.
    """

    @property
    @abstractmethod
    def type(self) -> str:
        """
        The type of the operator in `SUPPORTED_OPS`.
        """
        pass

    @abstractmethod
    def prepare(self, mesh: Mesh):
        """
        Prepare the operator for discretization.
        """
        pass

    @abstractmethod
    def run(self, source: Field) -> Field | LinearEqs:
        """
        Run the operator on each element.
        """
        pass
