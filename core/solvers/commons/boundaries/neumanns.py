# -*- encoding: utf-8 -*-
"""
Copyright (C) 2025, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

To provide the Neumann boundary condition.
"""
from core.solvers.interfaces import IBoundaryCondition, BoundaryType
from core.numerics.mesh import Element
from core.numerics.fields import Variable
from configs.settings import logger


class NeumannBoundary(IBoundaryCondition):
    """
    Neumann boundary condition, providing the flux at boundary.
    """

    @classmethod
    def get_name(cls) -> str:
        return "neumann"

    @classmethod
    def get_type(cls) -> BoundaryType:
        return BoundaryType.NATURAL

    def __init__(self, id: str, flux: Variable):
        """
        Initialize the custom boundary condition.

        Args:
            id: The unique identifier.
            value: The specified boundary value.
        """
        self._id = id
        self._flux = flux

    @property
    def id(self) -> str:
        return self._id

    def update(self, flux: Variable):
        """
        Update the boundary condition.

        Args:
            value: New boundary value.
        """
        self._flux = flux
        logger.info(f"Boundary condition {self.id} updated.")

    def evaluate(self) -> tuple:
        """
        Evaluate the boundary condition.

        Returns:
            The value.
        """
        return None, self._flux, None
