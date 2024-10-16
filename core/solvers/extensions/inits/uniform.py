# -*- encoding: utf-8 -*-
"""
Copyright (C) 2024, The YunmengEnvs Contributors. Join us, for you talents!  

Initialization by uniform value method.
"""
from core.solvers.interfaces import IInitCondition
from core.numerics.fields import Variable, Field


class UniformInitialization(IInitCondition):
    """
    Uniform initialization condition.
    """

    @classmethod
    def get_name(cls) -> str:
        return "uniform"

    def __init__(self, id: str, value: Variable):
        """
        Initialize the uniform initialization condition.

        Args:
            id: The identifier.
            value: The value used for initialization.
        """
        self._id = id
        self._value = value

    @property
    def id(self) -> str:
        return self._id

    def apply(self, field: Field) -> None:
        field.assign(self._value)
