# -*- encoding: utf-8 -*-
"""
Copyright (C) 2024, The YunMengEnvs Project Contributors. Join us !
License: Apache License 2.0

Interface for unit.
"""
from abc import abstractmethod
from typing import Optional

from core.solutions.standards.IDescribable import IDescribable
from core.solutions.standards.IDimension import IDimension


class IUnit(IDescribable):
    """Unit describes the physical unit."""

    @property
    @abstractmethod
    def dimension(self) -> Optional[IDimension]:
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
