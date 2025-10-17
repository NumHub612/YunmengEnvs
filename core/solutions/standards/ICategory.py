# -*- encoding: utf-8 -*-
"""
Copyright (C) 2024, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Interface for describing categories.
"""
from core.solutions.standards.IDescribable import IDescribable

from dataclasses import dataclass
from typing import Any


@dataclass
class ICategory(IDescribable):
    """Class describes one item of a possible categorization.

    It is used by the `IQuality` interface for describing qualitative data.
    A category defines one "class" within a "set of classes".
    """

    # Category name
    value: Any = None
