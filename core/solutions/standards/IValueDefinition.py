# -*- encoding: utf-8 -*-
"""
Copyright (C) 2024, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Interface for value definition.
"""
from core.solutions.standards.IDescribable import IDescribable

from abc import abstractmethod
from typing import Any


class IValueDefinition(IDescribable):
    """Class describes value definition."""

    @property
    @abstractmethod
    def value_type(self) -> type:
        """Value type."""
        pass

    @property
    @abstractmethod
    def missing_data_value(self) -> Any:
        """Missing data value."""
        pass
