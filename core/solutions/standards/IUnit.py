# -*- encoding: utf-8 -*-
"""
Copyright (C) 2024, The YunMengEnvs Project Contributors. Join us !
License: Apache License 2.0

Interface for unit.
"""
from abc import abstractmethod
from typing import Optional

from .IDescribable import IDescribable
from .IDimension import IDimension


class IUnit(IDescribable):
    """Unit describes the physical unit."""

    @abstractmethod
    def get_dimension(self) -> Optional[IDimension]:
        """Return the dimension of the unit."""
        pass

    @abstractmethod
    def get_conversion_factor_to_si(self) -> float:
        """Get the conversion factor to SI unit."""
        pass

    @abstractmethod
    def get_offset_to_si(self) -> float:
        """Get the offset to SI unit."""
        pass
