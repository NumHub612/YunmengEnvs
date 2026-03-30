# -*- encoding: utf-8 -*-
"""
Copyright (C) 2025, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

To provide the Neumann boundary condition.
"""
from core.solvers.interfaces import IBoundaryCondition, BoundaryType, BoundaryValue
from core.numerics.fields.variables import Variable, Var
from configs.settings import logger


class FluxBoundary(IBoundaryCondition):
    """
    Flux boundary condition, providing the flux at boundary.
    """

    @classmethod
    def get_type(cls) -> BoundaryType:
        return BoundaryType.FLUX

    @classmethod
    def get_name(cls) -> str:
        return "neumann"

    def __init__(self, id: str, flux: float | list[float]):
        self._id = id
        self._bc = BoundaryValue(flux=Var(flux))

    @property
    def id(self) -> str:
        return self._id

    def update(self, flux: Variable):
        """
        Update the boundary condition.

        Args:
            value: New boundary value.
        """
        if flux.type != self._bc.flux.type:
            raise ValueError(
                f"The new flux must have the same type {flux.type} "
                f"as the current flux {self._bc.flux.type}."
            )
        self._bc.flux = flux
        logger.info(f"Boundary {self.id} updated to {flux}.")

    def evaluate(self) -> BoundaryValue:
        return self._bc
