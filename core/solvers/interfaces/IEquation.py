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
                - description (str): Brief description of the variable.
                - output (bool): Whether to output the variable.
                - bounds (tuple): Bounds of the variable.
        """
        pass

    @abstractmethod
    def set_coefficients(self, coefficients: dict):
        """
        Set the coefficients for the equations.

        Args:
            coefficients: Dictionary of coefficients fields.
        """
        pass

    @abstractmethod
    def get_variables(self) -> dict:
        """
        Get the variables used in equations.
        """
        pass

    @abstractmethod
    def discretize(self, mesh: "Mesh", **kwargs):
        """
        Discretize the equation system.

        Args:
            mesh: The domain mesh.
        """
        pass

    @abstractmethod
    def update_interior(
        self,
        element: "Node | Face | Cell",
        neighbors: list,
        **kwargs,
    ) -> "Variable":
        """
        Update the interior element at the domain.

        Args:
            element: The interior element to be updated.
            neighbors: The neighboring elements.

        Returns:
            The updated element variable.
        """
        pass

    @abstractmethod
    def update_boundary(
        self,
        element: "Node | Face | Cell",
        boundary_conditions: tuple,
        neighbors: list,
        **kwargs,
    ) -> "Variable":
        """
        Update the boundary element at the domain.

        Args:
            element: The boundary element to be updated.
            boundary_conditions: Boundary conditions.
            neighbors: The neighboring elements.

        Returns:
            The updated element variable.
        """
        pass

    @abstractmethod
    def update(self, boundary_conditions: dict, **kwargs) -> dict:
        """
        Update the fields of the equations.

        Args:
            boundary_conditions: Boundary conditions.

        Returns:
            The updated fields.
        """
        pass


class IOperator(ABC):
    """
    Interface for discretizing pde item to a numerical form.
    """

    @property
    @abstractmethod
    def type(self) -> str:
        """
        The type of the operator.
        """
        pass

    @abstractmethod
    def prepare(
        self,
        mesh: "Mesh",
        field: "Field",
        coefficents: "Variable | Field",
        **kwargs,
    ):
        """
        Discretize the operator.

        Args:
            mesh: The domain mesh.
            field: The field to be discretized.
            coefficents: The coefficients of the operator.
        """
        pass

    @abstractmethod
    def run(self, element: int, neighbors: list, **kwargs) -> "Variable":
        """
        Run the operator.

        Args:
            element: The id of the element to be updated.
            neighbors: Ids of the neighboring elements.

        Returns:
            The operator result at the element.
        """
        pass

    @abstractmethod
    def get_result(self) -> "Field":
        """
        Get the operator result.
        """
        pass
