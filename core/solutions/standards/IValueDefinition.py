# -*- encoding: utf-8 -*-
"""
Copyright (C) 2024, The YunmengEnvs Contributors. Join us, for you talents!  
URL: https://github.com/NumHub612/YunmengEnvs  
License: Apache License 2.0

Interface for value definition.
"""
from abc import abstractmethod
from typing import Any

from core.solutions.standards.IDescribable import IDescribable


class IValueDefinition(IDescribable):
    """Class describes value definition."""

    @property
    @abstractmethod
    def value_type(self) -> Any:
        """Value type."""
        pass

    @property
    @abstractmethod
    def missing_data_value(self) -> Any:
        """Missing data value."""
        pass
