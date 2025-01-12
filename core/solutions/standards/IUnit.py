# -*- encoding: utf-8 -*-
"""
Copyright (C) 2024, The YunmengEnvs Contributors. Join us, share your ideas!  
 
Interface for unit.
"""
from core.solutions.standards.IDescribable import IDescribable
from core.solutions.standards.IDimension import IDimension

from abc import abstractmethod


class IUnit(IDescribable):
    """Unit describes the physical unit."""

    @property
    @abstractmethod
    def dimension(self) -> IDimension:
        """Return the dimension of the unit."""
        pass

    @property
    @abstractmethod
    def conversion_factor_to_si(self) -> float:
        """Get the conversion factor to SI unit."""
        pass

    @property
    @abstractmethod
    def offset_to_si(self) -> float:
        """Get the offset to SI unit."""
        pass
