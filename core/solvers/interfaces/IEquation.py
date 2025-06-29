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

    @abstractmethod
    def set_equations(self, equations: list, symbols: dict):
        """
        Set the equations and symbols for the equation.

        Args:
            equations: A list of equations expressions.
            symbols: Symbols used in the equations.

        Notes:
            - `symbols` discribe the variables as:
                - description (str): A brief description.
                - type (str): Type of the variable, e.g. "scalar", "vector", "tensor"
                - coefficent (bool): Whether the variable is coefficient.
                - bounds (tuple): Bounds of the variable.
        """
        pass

    @abstractmethod
    def set_coefficients(self, coefficients: dict):
        """
        Set the coefficients for the equations.

        Args:
            coefficients: Coefficients for the equations.
        """
        pass

    @abstractmethod
    def set_fields(self, fields: dict):
        """
        Set the variable fields.

        Args:
            fields: Dictionary of the fields.
        """
        pass

    @abstractmethod
    def set_mesh(self, mesh: Mesh):
        """
        Set the domain mesh.

        Args:
            mesh: The domain mesh.
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
        """The type of the operator."""
        pass

    @abstractmethod
    def prepare(self, mesh: Mesh):
        """
        Prepare the operator for running discretization.
        """
        pass

    @abstractmethod
    def run(self, source: Field) -> Field | LinearEqs:
        """
        Run the operator on each element.

        Args:
            source: The source field to be discretized.

        Returns:
            The discretized results.
        """
        pass
