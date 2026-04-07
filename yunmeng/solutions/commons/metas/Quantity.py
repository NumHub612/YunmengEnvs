# -*- encoding: utf-8 -*-
"""
Copyright (C) 2025, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Quantities and Units.
"""
from yunmeng.solutions.standards import IQuantity
from yunmeng.solutions.standards import IUnit, IDimension, DimensionBase
from yunmeng.numerics.fields import Variable
from enum import Enum

from dataclasses import dataclass
from typing import Any


class PredifinedDimensions(Enum):
    """Predefined dimensions."""

    LENGTH = IDimension.from_dict({DimensionBase.LENGTH: 1})
    AREA = IDimension.from_dict({DimensionBase.LENGTH: 2})
    VOLUME = IDimension.from_dict({DimensionBase.LENGTH: 3})
    MASS = IDimension.from_dict({DimensionBase.MASS: 1})
    TIME = IDimension.from_dict({DimensionBase.TIME: 1})

    MOMENTUM_FLUX = IDimension.from_dict(
        {
            DimensionBase.MASS: 1,
            DimensionBase.LENGTH: -1,
            DimensionBase.TIME: -2,
        }
    )
    ENERGY_FLUX = IDimension.from_dict(
        {
            DimensionBase.MASS: 2,
            DimensionBase.LENGTH: -2,
            DimensionBase.TIME: -1,
        }
    )
    MASS_FLUX = IDimension.from_dict(
        {
            DimensionBase.MASS: 1,
            DimensionBase.LENGTH: -2,
            DimensionBase.TIME: -1,
        }
    )


class PredefinedUnits(Enum):
    """Predefined units commonly used."""

    METER = IUnit(
        "m",
        "meter",
        PredifinedDimensions.LENGTH.value,
        1.0,
        0.0,
    )
    LITER = IUnit(
        "L",
        "liter",
        PredifinedDimensions.VOLUME.value,
        0.001,
        0.0,
    )
    AREA = IUnit(
        "m2",
        "square meter",
        PredifinedDimensions.AREA.value,
        1.0,
        0.0,
    )
    DISCHARGE = IUnit(
        "m3/s",
        "cubic meter per second",
        IDimension.from_dict(
            {
                DimensionBase.LENGTH: 3,
                DimensionBase.TIME: -1,
            }
        ),
        1.0,
        0.0,
    )
    DENSITY = IUnit(
        "kg/m3",
        "mass per cubic meter",
        IDimension.from_dict(
            {
                DimensionBase.MASS: 1,
                DimensionBase.LENGTH: -3,
            }
        ),
        1.0,
        0.0,
    )
    CONCENTRATION = IUnit(
        "mg/L",
        "mass per cubic meter",
        IDimension.from_dict(
            {
                DimensionBase.MASS: 1,
                DimensionBase.LENGTH: -3,
            }
        ),
        1.0,
        0.0,
    )
    VELOCITY = IUnit(
        "m/s",
        "velocity",
        IDimension.from_dict(
            {
                DimensionBase.LENGTH: 1,
                DimensionBase.TIME: -1,
            }
        ),
        1.0,
        0.0,
    )
    MILLIMETER_PER_DAY = IUnit(
        "mm/d",
        "millimeter per day",
        IDimension.from_dict(
            {
                DimensionBase.LENGTH: 1,
                DimensionBase.TIME: -1,
            }
        ),
        1.15741e-08,
        0.0,
    )
    MILLIMETER_PER_HOUR = IUnit(
        "mm/h",
        "millimeter per hour",
        IDimension.from_dict(
            {
                DimensionBase.LENGTH: 1,
                DimensionBase.TIME: -1,
            }
        ),
        0.000277778,
        0.0,
    )


@dataclass
class Quantity(IQuantity):
    """Quantity class."""

    def __init__(
        self,
        value: float | bool | Variable,
        unit: IUnit = None,
        caption: str = "",
        description: str = "",
        missing_value: Any = None,
    ):
        super().__init__(caption, description, None, missing_value)
        self.value_type = type(value)
        self.value = value
        self.unit = unit

    def to_si(self) -> Any:
        """Convert the value to SI unit."""
        if self.unit is None:
            return self.value

        return self.value * self.unit.conversion + self.unit.offset

    def is_missing(self) -> bool:
        """Check if the value missing."""
        if self.value is None:
            return True
        if self.value == self.missing_data_value:
            return True
        return False
