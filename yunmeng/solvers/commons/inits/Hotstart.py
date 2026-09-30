# -*- encoding: utf-8 -*-
"""
Copyright (C) 2026, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Hotstart initial condition.
"""

from yunmeng.interfaces.types import ArrayLike
from yunmeng.interfaces.supports import IField
from yunmeng.interfaces.solver import IInitialCondition
from yunmeng.numerics.algos import ym_register


@ym_register("init")
class HotstartIC(IInitialCondition):
    """Hotstart initial condition directly specified the field value."""

    def __init__(
        self,
        ic_id: str,
        target_field: str,
        value: float | ArrayLike | IField,
    ):
        self._id = ic_id
        self._field = target_field
        self._value = value

    @classmethod
    def get_name(cls) -> str:
        return "HotstartIC"

    @property
    def id(self) -> str:
        return self._id

    @property
    def target_field(self) -> str:
        return self._field

    def apply(self, field: IField):
        if field.meta.name != self._field:
            raise ValueError(
                f"Field {field.meta.name} doesn't match the IC {self.id} target field."
            )
        if isinstance(self._value, IField):
            if self._value.size != field.size:
                raise ValueError(
                    f"Field {field.meta.name} size doesn't match the IC {self.id}."
                )
            field.values[:] = self._value.values
        else:
            field.values[:] = self._value
