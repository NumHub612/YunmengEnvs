# -*- encoding: utf-8 -*-
"""
Copyright (C) 2024, The YunmengEnvs Contributors. Join us, for you talents!  

Initialization by hot-starting field.
"""
from core.solvers.interfaces import IInitCondition
from core.numerics.fields import Field


class HotstartInitialization(IInitCondition):
    """
    Hotstart initialization condition.
    """

    @classmethod
    def get_name(cls) -> str:
        return "hotstart"

    def __init__(self, id: str, field: Field):
        """
        Initialize the initialization method.

        Args:
            id: The identifier.
            field: The field used for initialization.
        """
        self._id = id
        self._field = field

    @property
    def id(self) -> str:
        return self._id

    def apply(self, field: Field):
        field.assign(self._field)
