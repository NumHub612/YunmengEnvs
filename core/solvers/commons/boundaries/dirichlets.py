# -*- encoding: utf-8 -*-
"""
Copyright (C) 2025, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

To provide the Dirichlet boundary condition.
"""
from core.solvers.interfaces import IBoundaryCondition, BoundaryType
from core.numerics.mesh import Element
from core.numerics.fields import Variable
from configs.settings import logger


class DirichletBoundary(IBoundaryCondition):
    """
    Dirichlet boundary condition, providing the variable value at boundary.
    """

    @classmethod
    def get_name(cls) -> str:
        return "dirichlet"

    @classmethod
    def get_type(cls) -> BoundaryType:
        return BoundaryType.FIXED

    def __init__(self, id: str, value: Variable):
        """
        Initialize the custom boundary condition.

        Args:
            id: The unique identifier.
            value: The specified boundary value.
        """
        self._id = id
        self._value = value

    @property
    def id(self) -> str:
        return self._id

    def update(self, value: Variable):
        """
        Update the boundary condition.

        Args:
            value: New boundary value.
        """
        self._value = value
        logger.info(f"Boundary condition {self.id} updated.")

    def evaluate(self) -> tuple:
        """
        Evaluate the boundary condition.

        Returns:
            The value.
        """
        return self._value, None, None
