# -*- encoding: utf-8 -*-
"""
Copyright (C) 2024, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Interfaces for initializing the solver fields.
"""
from yunmeng.numerics.fields.fields import Field
from abc import ABC, abstractmethod


class IInitCondition(ABC):
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
    def id(self) -> str:
        """
        The instance id.
        """
        pass

    @abstractmethod
    def apply(self, target_field: Field):
        """
        Initializes the target field in-place.
        """
        pass
