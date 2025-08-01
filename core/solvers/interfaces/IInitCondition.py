# -*- encoding: utf-8 -*-
"""
Copyright (C) 2024, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Interfaces for initializing the firield variables.
"""
from core.numerics.fields import Field
from abc import ABC, abstractmethod


class IInitCondition(ABC):
    """
    Interface class for initializing field.
    """

    @classmethod
    @abstractmethod
    def get_name(cls) -> str:
        """
        The unique name of this method.
        """
        pass

    @property
    @abstractmethod
    def id(self) -> str:
        """
        The initial condition id.
        """
        pass

    @abstractmethod
    def apply(self, field: Field):
        """
        Initializes the field variables.
        """
        pass
