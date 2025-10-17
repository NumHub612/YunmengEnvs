# -*- encoding: utf-8 -*-
"""
Copyright (C) 2024, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Interface for unit.
"""
from core.solutions.standards.IDescribable import IDescribable
from core.solutions.standards.IDimension import IDimension

from dataclasses import dataclass


@dataclass
class IUnit(IDescribable):
    """Unit describes the physical unit."""

    # Dimension of the unit.
    dimension: IDimension = None

    # Conversion factor to SI unit.
    conversion_factor_to_si: float = 1.0

    # Offset to SI unit.
    offset_to_si: float = 0.0
