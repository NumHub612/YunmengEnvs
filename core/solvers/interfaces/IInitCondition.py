# -*- encoding: utf-8 -*-
"""
Copyright (C) 2024, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Interfaces for initializing the firield variables.
"""
from abc import ABC, abstractmethod
from core.numerics.fields import Field


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
    def apply(self, field: Field) -> None:
        """
        Initializes the field variables.

        Args:
            field: Field to be initialized.
        """
        pass
