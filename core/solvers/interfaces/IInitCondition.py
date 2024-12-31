# -*- encoding: utf-8 -*-
"""
Copyright (C) 2024, The YunmengEnvs Contributors. Join us, share your ideas!  

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
        The unique name of the initialization method.
        """
        pass

    @abstractmethod
    def apply(self, field: Field, **kwargs) -> None:
        """
        Initializes the field variables.

        Args:
            field: Field to be initialized.
        """
        pass
