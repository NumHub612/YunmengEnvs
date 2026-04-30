# -*- encoding: utf-8 -*-
"""
Copyright (C) 2024, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Initialization by hot-starting field.
"""
from yunmeng.solvers.interfaces import IInitCondition
from yunmeng.numerics.fields.fields import Field


class HotstartInitialization(IInitCondition):
    """
    Hotstart initialization condition.
    """

    @classmethod
    def get_name(cls) -> str:
        return "hotstart"

    def __init__(self, id: str, src_field: Field):
        self._id = id
        self._src_field = src_field

    @property
    def id(self) -> str:
        return self._id

    def apply(self, target_field: Field):
        if target_field.vtype != self._src_field.vtype:
            raise ValueError(
                f"The hotstart field must have the same vtype {self._src_field.vtype} "
                f"as the target field {target_field.vtype}."
            )
        if target_field.size != self._src_field.size:
            raise ValueError(
                f"The hotstart field must have the same size {self._src_field.size} "
                f"as the target field {target_field.size}."
            )

        local_field = self._src_field.gather_to_host()
        target_field.scatter_from_host(local_field)
