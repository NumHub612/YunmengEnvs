# -*- encoding: utf-8 -*-
"""
Copyright (C) 2024, The YunmengEnvs Contributors. Join us, share your ideas!  
 
Interface for quantity values.
"""
from core.solutions.standards.IValueDefinition import IValueDefinition
from core.solutions.standards.IUnit import IUnit

from abc import abstractmethod
from typing import Optional


class IQuantity(IValueDefinition):
    """
    Class specifies values as an amount of some unit.
    """

    @abstractmethod
    def get_unit(self) -> Optional[IUnit]:
        """Unit of quantity."""
        pass
