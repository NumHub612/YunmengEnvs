# -*- encoding: utf-8 -*-
"""
Copyright (C) 2024, The YunmengEnvs Contributors. Join us, for you talents!  

Interfaces for initializing the firield variables.
"""
from abc import ABC, abstractmethod

from core.numerics.fields import Field


class IInitCondition(ABC):
    """
    Interface for initializing a fields.
    """

    @classmethod
    @abstractmethod
    def get_name(cls) -> str:
        """
        The unique name of the initialization method.
        """
        raise NotImplementedError

    @abstractmethod
    def apply_to(self, field: Field) -> None:
        """
        Initializes the field variables.
        """
        raise NotImplementedError
