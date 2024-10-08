# -*- encoding: utf-8 -*-
"""
Copyright (C) 2024, The YunmengEnvs Contributors. Join us, for you talents!  
 
Interface for quantity values.
"""
from abc import abstractmethod
from typing import Optional

from core.solutions.standards.IUnit import IUnit
from core.solutions.standards.IValueDefinition import IValueDefinition


class IQuantity(IValueDefinition):
    """
    Class specifies values as an amount of some unit.
    """

    @abstractmethod
    def get_unit(self) -> Optional[IUnit]:
        """Unit of quantity."""
        pass
