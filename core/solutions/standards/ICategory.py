# -*- encoding: utf-8 -*-
"""
Copyright (C) 2024, The YunMengEnvs Project Contributors. Join us !
License: Apache License 2.0

Interface for describing categories.
"""
from abc import abstractmethod
from typing import Any

from core.solutions.standards.IDescribable import IDescribable


class ICategory(IDescribable):
    """Class describes one item of a possible categorization.

    It is used by the `IQuality` interface for describing qualitative data. A category
    defines one "class" within a "set of classes".
    """

    @property
    @abstractmethod
    def value(self) -> Any:
        """Value for this category."""
        pass
