# -*- encoding: utf-8 -*-
"""
Copyright (C) 2024, The YunmengEnvs Contributors. Join us, for you talents!  

Constant boundary condition.
"""
from core.solvers.interfaces import IBoundaryCondition
from core.numerics.mesh import Node, Face, Cell
from core.numerics.fields import Variable

from typing import Callable


class ConstantBoundary(IBoundaryCondition):
    """
    Constant boundary condition.
    """

    @classmethod
    def get_name(cls) -> str:
        return "constant"

    def __init__(self, id: str, value: Variable, flux: Variable):
        """
        Initialize the boundary condition.

        Args:
            id: The unique identifier.
            value: The value of the variable at the boundary.
            flux: The flux of the variable at the boundary.
        """
        self._id = id
        self._value = value
        self._flux = flux

    @property
    def id(self) -> str:
        return self._id

    def evaluate(self, t: float, elem: Node | Face | Cell) -> tuple:
        return self._flux, self._value
