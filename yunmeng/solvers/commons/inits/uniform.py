# -*- encoding: utf-8 -*-
"""
Copyright (C) 2024, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Initialization by uniform value method.
"""

from yunmeng.solvers.interfaces import IInitialCondition
from yunmeng.numerics.fields import Field, Var
import numpy as np


class UniformInitialization(IInitialCondition):
    """
    Uniform initialization condition.
    """

    @classmethod
    def get_name(cls) -> str:
        return "uniform"

    def __init__(self, id: str, value: float | list[float]):
        self._id = id
        self._value = Var(value)

    @property
    def id(self) -> str:
        return self._id

    def apply(self, target_field: Field):
        if target_field.vtype != self._value.type:
            raise ValueError(
                f"The uniform init value must have the same type {self._value.type} "
                f"as the target field {target_field.vtype}."
            )
        full_data = np.full(target_field.size, self._value.data, dtype=np.float64)
        target_field.scatter_from_host(full_data)
