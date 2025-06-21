# -*- encoding: utf-8 -*-
"""
Copyright (C) 2024, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Interfaces for boundary conditions at faces of a mesh.
"""
from core.numerics.mesh import Element
from abc import ABC, abstractmethod


class IBoundaryCondition(ABC):
    """
    Interface for pde boundary conditions classes.
    """

    @classmethod
    @abstractmethod
    def get_name(cls) -> str:
        """
        The unique name of the boundary condition.
        """
        pass

    @abstractmethod
    def update(self, time: float):
        """
        Update the boundary condition parameters.
        Used for running online.

        Args:
            time: The current time.
        """
        pass

    @abstractmethod
    def evaluate(self, time: float, element: Element) -> tuple:
        """
        Evaluate the boundary condition.

        Args:
            time: The current time.
            element: The mesh element to be evaluated.

        Returns:
            Tuple of boundary values, where:
            (flux, value, *extras).
        """
        pass
