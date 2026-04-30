# -*- encoding: utf-8 -*-
"""
Copyright (C) 2024, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Interface for unit.
"""
from yunmeng.solutions.standards.IDescribable import IDescribable
from yunmeng.solutions.standards.IDimension import IDimension

from dataclasses import dataclass


@dataclass
class IUnit(IDescribable):
    """Unit describes the physical unit."""

    # Dimension of the unit.
    dimension: IDimension = None

    # Conversion factor to SI unit.
    conversion: float = 1.0

    # Offset to SI unit.
    offset: float = 0.0
