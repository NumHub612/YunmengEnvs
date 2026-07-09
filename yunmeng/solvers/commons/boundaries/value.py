# -*- encoding: utf-8 -*-
"""
Copyright (C) 2025, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

To provide the Dirichlet boundary condition.
"""

from yunmeng.solvers.commons.solvers import BaseBoundary, BoundaryType, BoundaryValue
from yunmeng.numerics.mesh import Region
from yunmeng.numerics.fields import Field, Variable, Var
from yunmeng.setting import logger


class ValueBoundary(BaseBoundary):
    """
    Value boundary condition provides the variable value at boundary.
    """

    @classmethod
    def get_type(cls) -> BoundaryType:
        return BoundaryType.VALUE

    @classmethod
    def get_name(cls) -> str:
        return "dirichlet"

    def __init__(
        self,
        id: str,
        target_field: str,
        region: Region,
        value: Variable | float | list[float],
    ):
        """
        Args:
            id: Unique identifier for this BC instance.
            target_field: Name of the field this BC applies to (e.g., "u").
            region: Region defining the boundary location.
            value: Prescribed boundary value.
        """
        super().__init__(id, target_field, region)
        self._bc = BoundaryValue(value=Var(value))

    def apply(self, field: Field):
        if field.meta.name != self._target_field:
            logger.warning(
                f"Field {field.meta.name} does not match target field {self._target_field}"
            )

        if field.meta.vtype != self._bc.value.vtype:
            raise ValueError(
                f"Field {field.meta.name} has variable type {field.meta.vtype}, "
                f"but BC {self._id} has variable type {self._bc.value.vtype}."
            )

        # TODO：check the gpu/cpu senario data exchange.
        field[self._resolved_ids] = self._bc.value.data
