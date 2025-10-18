# -*- encoding: utf-8 -*-
"""
Copyright (C) 2024, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Interface for value definition.
"""
from core.solutions.standards.IDescribable import IDescribable

from dataclasses import dataclass
from typing import Any


@dataclass
class IValueDefinition(IDescribable):
    """Class describes value definition."""

    # Value type.
    value_type: type = None

    # Flag representing missing data.
    missing_value: Any = None
