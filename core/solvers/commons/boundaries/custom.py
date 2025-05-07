# -*- encoding: utf-8 -*-
"""
Copyright (C) 2024, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Custom defined boundary condition.
"""
from core.solvers.interfaces import IBoundaryCondition
from core.numerics.mesh import Element
from configs.settings import logger

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

    def update(self, time: float, bc_func: Callable):
        """
        Update the boundary condition function.

        Args:
            bc_func: The new function.
        """
        self._bc_func = bc_func
        logger.info(f"Boundary condition {self._id} updated at time {time}.")

    def evaluate(self, time: float, element: Element) -> tuple:
        flux, value = self._bc_func(time, element)
        return flux, value
