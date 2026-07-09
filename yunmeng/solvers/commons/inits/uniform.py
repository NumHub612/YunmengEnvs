# -*- encoding: utf-8 -*-
"""
Copyright (C) 2024, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Initialization by uniform value method.
"""

from yunmeng.solvers.commons.solvers import BaseInitializer
from yunmeng.numerics.fields import Field, Variable, Var
import numpy as np


class UniformInitializer(BaseInitializer):
    """
    Uniform initialization condition.
    """

    @classmethod
    def get_name(cls) -> str:
        return "uniform"

    def __init__(self, id: str, target_field: str, value: float | list[float]):
        super().__init__(id, target_field)
        self._value = Var(value)

    def get(self) -> Variable:
        return self._value

    def apply(self, field: Field):
        if field.vtype != self._value.vtype:
            raise ValueError(
                f"The uniform init value must have the same type {self._value.vtype} "
                f"as the target field {field.vtype}."
            )
        full_data = np.full(field.size, self._value.data, dtype=np.float64)
        field.scatter_from_host(full_data)
