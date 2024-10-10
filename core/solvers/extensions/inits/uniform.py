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

    def __init__(self, value: Variable):
        self._value = value

    @classmethod
    def get_name(cls) -> str:
        return "uniform"

    def apply_to(self, field: Field):
        field.assign(self._value)
