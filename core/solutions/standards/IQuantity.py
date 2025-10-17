# -*- encoding: utf-8 -*-
"""
Copyright (C) 2024, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Interface for quantity values.
"""
from core.solutions.standards.IValueDefinition import IValueDefinition
from core.solutions.standards.IUnit import IUnit

from dataclasses import dataclass
from typing import Any


@dataclass
class IQuantity(IValueDefinition):
    """
    Class specifies values as an amount of units.
    """

    # Numeric value of quantity.
    value: Any = None

    # Unit of quantity.
    unit: IUnit = None
