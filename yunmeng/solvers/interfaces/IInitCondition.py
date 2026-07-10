# -*- encoding: utf-8 -*-
"""
Copyright (C) 2024, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Interfaces for initializing the solver fields.
"""

from abc import ABC, abstractmethod
from yunmeng.numerics.fields import Field, Variable


class IInitialCondition(ABC):
    """
    Interface class for initializing the field.
    """

    @classmethod
    @abstractmethod
    def get_name(cls) -> str:
        """
        The unique name of the method.
        """
        pass

    @property
    @abstractmethod
    def target_field(self) -> str:
        """
        The target field.
        """
        pass

    @property
    @abstractmethod
    def id(self) -> str:
        """
        The instance id.
        """
        pass

    @abstractmethod
    def get(self, **kwargs) -> Variable:
        """
        Get the initial value.
        """
        pass

    @abstractmethod
    def apply(self, field: Field):
        """
        Initializes the target field.
        """
        pass
