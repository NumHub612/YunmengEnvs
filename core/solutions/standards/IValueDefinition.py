# -*- encoding: utf-8 -*-
"""
Copyright (C) 2024, The YunMengEnvs Project Contributors. Join us !
License: Apache License 2.0

Interface for value definition.
"""
from abc import abstractmethod
from typing import Any

from .IDescribable import IDescribable


class IValueDefinition(IDescribable):
    """Class describes value definition."""

    @abstractmethod
    def get_value_type(self) -> Any:
        """Get value type."""
        pass

    @abstractmethod
    def get_missing_data_value(self) -> Any:
        """Get missing data value."""
        pass
