# -*- encoding: utf-8 -*-
"""
Copyright (C) 2025, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

To provide the Robinn boundary condition.
"""
from yunmeng.solvers.interfaces import IBoundaryCondition, BoundaryType, BoundaryValue
from yunmeng.numerics.fields.variables import Variable, Var
from yunmeng.setting import logger


class MixedBoundary(IBoundaryCondition):
    """
    Mixed boundary condition presents linear combination of value and flux.
    Required:
    + value,
    + flux,
    + extra={'coeff_value': float, 'coeff_flux': float}
    """

    @classmethod
    def get_type(cls) -> BoundaryType:
        return BoundaryType.MIXED

    @classmethod
    def get_name(cls) -> str:
        return "robinn"

    def __init__(
        self,
        id: str,
        value: float | list[float],
        flux: float | list[float],
        coeff_value: float = 1.0,
        coeff_flux: float = 1.0,
    ):
        self._id = id
        self._bc = BoundaryValue(
            value=Var(value),
            flux=Var(flux),
            extra={"coeff_value": coeff_value, "coeff_flux": coeff_flux},
        )

    @property
    def id(self) -> str:
        return self._id

    def update(
        self,
        value: Variable,
        flux: Variable,
        coeff_value: float = None,
        coeff_flux: float = None,
    ):
        if value.type != self._bc.value.type:
            raise ValueError(
                f"The new value must have the same type {value.type} "
                f"as the current value {self._bc.value.type}."
            )
        if flux.type != self._bc.flux.type:
            raise ValueError(
                f"The new flux must have the same type {flux.type} "
                f"as the current flux {self._bc.flux.type}."
            )
        self._bc.value = value
        self._bc.flux = flux
        if coeff_value is not None:
            self._bc.extra["coeff_value"] = coeff_value
        if coeff_flux is not None:
            self._bc.extra["coeff_flux"] = coeff_flux
        logger.info(f"Boundary {self.id} updated to {self._bc}.")

    def evaluate(self) -> tuple:
        return self._bc
