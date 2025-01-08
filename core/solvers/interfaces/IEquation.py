# -*- encoding: utf-8 -*-
"""
Copyright (C) 2024, The YunmengEnvs Contributors. Join us, share your ideas!  

Interface for describing and discretizing pde equations.
"""
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
    def set_mesh(self, mesh: "Mesh"):
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
    def discretize(self) -> "LinearEqs":
        """
        Discretize the equation system.

        Returns:
            The discretized result.
        """
        pass


class IOperator(ABC):
    """
    Interface for discretizing pde term to a numerical form.
    """

    @abstractmethod
    def prepare(self, mesh: "Mesh", coefficents: dict):
        """
        Prepare the operator for running.
        """
        pass

    @abstractmethod
    def run(self, element: int, neighbors: list[int]) -> "Variable | LinearEqs":
        """
        Run the operator on a given element.

        Args:
            element: The element id.
            neighbors: The ids of the neighboring elements.

        Returns:
            The result of the operator.
        """
        pass

    @property
    @abstractmethod
    def type(self) -> str:
        """
        The type of the operator.
        """
        pass

    @property
    @abstractmethod
    def scheme(self) -> str:
        """
        The operator scheme.
        """
        pass
