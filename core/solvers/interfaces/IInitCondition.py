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

    @property
    @abstractmethod
    def id(self) -> str:
        """
        The unique id of a initialization method instance.
        """
        pass

    @classmethod
    @abstractmethod
    def get_name(cls) -> str:
        """
        The unique name of the initialization method.
        """
        pass

    @abstractmethod
    def apply(self, field: Field, **kwargs) -> None:
        """
        Initializes the field variables.

        Args:
            field: The field to be initialized.
            kwargs: Additional arguments for initializing.
        """
        pass
