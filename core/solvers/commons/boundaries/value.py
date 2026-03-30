# -*- encoding: utf-8 -*-
"""
Copyright (C) 2025, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

To provide the Dirichlet boundary condition.
"""
from core.solvers.interfaces import IBoundaryCondition, BoundaryType, BoundaryValue
from core.numerics.fields.variables import Variable, Var
from configs.settings import logger


class ValueBoundary(IBoundaryCondition):
    """
    Value boundary condition provides the variable value at boundary.
    """

    @classmethod
    def get_type(cls) -> BoundaryType:
        return BoundaryType.VALUE

    @classmethod
    def get_name(cls) -> str:
        return "dirichlet"

    def __init__(self, id: str, value: float | list[float]):
        self._id = id
        self._bc = BoundaryValue(value=Var(value))

    @property
    def id(self) -> str:
        return self._id

    def update(self, value: Variable):
        """
        Update the boundary condition.

        Args:
            value: New boundary value.
        """
        if value.type != self._bc.value.type:
            raise ValueError(
                f"The new value must have the same type {value.type} "
                f"as the current value {self._bc.value.type}."
            )
        self._bc.value = value
        logger.info(f"Boundary {self.id} updated to {value}.")

    def evaluate(self) -> BoundaryValue:
        return self._bc
