# -*- encoding: utf-8 -*-
"""
Copyright (C) 2024, The YunMengEnvs Project Contributors. Join us !
License: Apache License 2.0

Interfaces for describing categories.
"""
from abc import abstractmethod

from .IDescribable import IDescribable


class ICategory(IDescribable):
    """Class describes one item of a possible categorization.

    It is used by the `IQuality` interface for describing qualitative data. A category
    defines one "class" within a "set of classes".
    """

    @abstractmethod
    def get_value(self):
        """Value for this category."""
        pass
