# -*- encoding: utf-8 -*-
"""
Copyright (C) 2024, The YunmengEnvs Contributors. Join us, share your ideas!  

Custom defined boundary condition.
"""
from core.solvers.interfaces import IBoundaryCondition
from core.numerics.mesh import Node, Face, Cell

from typing import Callable


class CustomBoundary(IBoundaryCondition):
    """
    Custom defined boundary condition.
    """

    @classmethod
    def get_name(cls) -> str:
        return "custom"

    def __init__(self, id: str, bc_func: Callable):
        """
        Initialize the custom boundary condition.

        Args:
            id: The unique identifier.
            bc_func: The function take time and element as input and return the boundary conditions.
        """
        self._id = id
        self._bc_func = bc_func

    @property
    def id(self) -> str:
        return self._id

    def evaluate(self, t: float, elem: Node | Face | Cell) -> tuple:
        flux, value = self._bc_func(t, elem)
        return flux, value
