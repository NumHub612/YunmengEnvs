# -*- encoding: utf-8 -*-
"""
Copyright (C) 2024, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Initialization by uniform value method.
"""
from core.solvers.interfaces import IInitCondition
from core.numerics.fields import Field, Var
import numpy as np


class UniformInitialization(IInitCondition):
    """
    Uniform initialization condition.
    """

    @classmethod
    def get_name(cls) -> str:
        return "uniform"

    def __init__(self, id: str, value: float | list[float]):
        """
        Initialize the uniform initialization condition.

        Args:
            id: The identifier.
            value: The value used for initialization.
        """
        self._id = id
        self._value = Var(value)

    @property
    def id(self) -> str:
        return self._id

    def apply(self, field: Field) -> None:
        field.assign(self._value)
