# -*- encoding: utf-8 -*-
"""
Copyright (C) 2024, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Initialization by hot-starting field.
"""

from yunmeng.solvers.commons.solvers import BaseInitializer
from yunmeng.numerics.fields import Field, Variable


class HotstartInitializer(BaseInitializer):
    """
    Hotstart initialization condition.
    """

    @classmethod
    def get_name(cls) -> str:
        return "hotstart"

    def __init__(self, id: str, target_field: str, src_field: Field):
        super().__init__(id, target_field)
        self._src_field = src_field

    def get(self, element_id: int) -> Variable:
        if element_id >= self._src_field.size:
            raise ValueError(
                f"element_id {element_id} is out of range of the hotstart field "
                f"with size {self._src_field.size}."
            )
        return self._src_field[element_id]

    def apply(self, field: Field):
        if field.vtype != self._src_field.vtype:
            raise ValueError(
                f"The hotstart field must have the same vtype {self._src_field.vtype} "
                f"as the target field {field.vtype}."
            )
        if field.size != self._src_field.size:
            raise ValueError(
                f"The hotstart field must have the same size {self._src_field.size} "
                f"as the target field {field.size}."
            )

        local_field = self._src_field.gather_to_host()
        field.scatter_from_host(local_field)
