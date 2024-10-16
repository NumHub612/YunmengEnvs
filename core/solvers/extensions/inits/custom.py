# -*- encoding: utf-8 -*-
"""
Copyright (C) 2024, The YunmengEnvs Contributors. Join us, for you talents!  

Initialization by custom method.
"""
from core.solvers.interfaces import IInitCondition
from core.numerics.fields import Field

from typing import Callable


class CustomInitialization(IInitCondition):
    """
    Custom defined initialization condition.
    """

    @classmethod
    def get_name(cls) -> str:
        return "custom"

    def __init__(self, id: str, init_func: Callable):
        """
        Initialize initialization condition.

        Args:
            id: The unique identifier.
            init_func: The initialization function taking a field as input and initializing it.
        """
        self._id = id
        self._init_func = init_func

    @property
    def id(self) -> str:
        return self._id

    def apply(self, field: Field) -> None:
        self._init_func(field)
