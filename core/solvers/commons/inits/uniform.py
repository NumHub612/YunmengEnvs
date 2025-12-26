# -*- encoding: utf-8 -*-
"""
Copyright (C) 2024, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Initialization by uniform value method.
"""
from core.solvers.interfaces import IInitCondition
from core.numerics.fields import Field, make_var_from_value
import numpy as np


class UniformInitialization(IInitCondition):
    """
    Uniform initialization condition.
    """

    @classmethod
    def get_name(cls) -> str:
        return "uniform"

    def __init__(
        self,
        id: str,
        value: float | list[float],
        dtype: str = "scalar",
    ):
        """
        Initialize the uniform initialization condition.

        Args:
            id: The identifier.
            value: The value used for initialization.
            dtype: The type of the variable.
        """
        self._id = id
        self._value = make_var_from_value(value, dtype)

    @property
    def id(self) -> str:
        return self._id

    def apply(self, field: Field) -> None:
        field.assign(self._value)
