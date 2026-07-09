# -*- encoding: utf-8 -*-
"""
Copyright (C) 2025, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

To provide the Neumann boundary condition.
"""

from yunmeng.solvers.commons.solvers import BaseBoundary, BoundaryType, BoundaryValue
from yunmeng.numerics.mesh import Region
from yunmeng.numerics.fields import Field, Var
from yunmeng.setting import logger


class FluxBoundary(BaseBoundary):
    """
    Flux boundary condition, providing the flux at boundary.
    """

    @classmethod
    def get_type(cls) -> BoundaryType:
        return BoundaryType.FLUX

    @classmethod
    def get_name(cls) -> str:
        return "neumann"

    def __init__(
        self,
        id: str,
        target_field: str,
        region: Region,
        flux: float | list[float],
    ):
        """
        Args:
            id: Unique identifier for this BC instance.
            target_field: Name of the field this BC applies to (e.g., "p").
            region: Region defining the boundary location.
            flux: Prescribed boundary flux.
        """
        super().__init__(id, target_field, region)
        self._bc = BoundaryValue(flux=Var(flux))

    def apply(self, field: Field):
        if field.meta.name != self._target_field:
            logger.warning(
                f"Field {field.meta.name} does not match target field {self._target_field}"
            )

        if field.meta.vtype != self._bc.value.vtype:
            raise ValueError(
                f"Field {field.meta.name} has variable type {field.meta.vtype}, "
                f"but BC {self._id} has variable type {self._bc.flux.vtype}."
            )

        field[self._resolved_ids] = self._bc.flux.data
