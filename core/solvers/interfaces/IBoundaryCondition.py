# -*- encoding: utf-8 -*-
"""
Copyright (C) 2024, The YunmengEnvs Contributors. Join us, for you talents!  

Interfaces for boundary conditions at faces of a mesh.
"""
from core.numerics.mesh import Node, Face, Cell

from abc import ABC, abstractmethod


class IBoundaryCondition(ABC):
    """
    Interface for boundary conditions.
    """

    @property
    @abstractmethod
    def id(self) -> str:
        """
        Unique id of a boundarycondition instance.
        """
        pass

    @classmethod
    @abstractmethod
    def get_name(cls) -> str:
        """
        The unique name of the boundary condition.
        """
        pass

    @abstractmethod
    def evaluate(self, time: float, elem: Node | Face | Cell, **kwargs) -> tuple:
        """
        Evaluate the boundary condition.

        Args:
            time: The current time, in seconds.
            elem: The element to be evaluated.
            kwargs: Additional arguments.

        Returns:
            A tuple of two values: flux and value.
        """
        pass
