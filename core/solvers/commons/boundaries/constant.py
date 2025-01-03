# -*- encoding: utf-8 -*-
"""
Copyright (C) 2024, The YunmengEnvs Contributors. Join us, share your ideas!  

Constant boundary condition.
"""
from core.solvers.interfaces import IBoundaryCondition
from core.numerics.mesh import Node, Face, Cell
from core.numerics.fields import Variable
from configs.settings import logger


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

    def update(self, time: float, value: Variable, flux: Variable):
        self._value = value
        self._flux = flux
        logger.info(
            f"Constant boundary condition {self.id} updated at \
                time {time} with value {value} and flux {flux}."
        )

    def evaluate(self, time: float, element: Node | Face | Cell) -> tuple:
        return self._flux, self._value
