# -*- encoding: utf-8 -*-
"""
Copyright (C) 2024, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

To provide dimensions management in physical quantities.
"""
from enum import Enum
from dataclasses import dataclass, field


class DimensionBase(Enum):
    """
    Base dimensions for physical quantities.
    """

    LENGTH = 0  # meters
    MASS = 1  # kilograms
    TIME = 2  # seconds
    ELECTRICCURRENT = 3  # amperes
    TEMPERATURE = 4  # kelvin
    AMOUNTOFSUBSTANCE = 5  # moles
    LUMINOUSINTENSITY = 6  # candelas
    CURRENCY = 7  # money
    UNITLESS = 8  # dimensionless


@dataclass
class IDimension:
    """To provide dimension-related operations."""

    powers: dict[DimensionBase, float] = field(default_factory=dict)

    def get_power(self, base_quantity: DimensionBase) -> float:
        """
        Gets the power for the requested dimension.
        """
        return self.powers.get(base_quantity, 0)

    def set_power(
        self,
        base_quantity: DimensionBase,
        power: float,
    ):
        """
        Sets a power for a base dimension.
        """
        self.powers[base_quantity] = power

    @staticmethod
    def from_dict(dimensions: dict[DimensionBase, float]) -> "IDimension":
        """
        Makes a new IDimension object with the given dimensions set.
        """
        dim = IDimension()
        for base_quantity, power in dimensions.items():
            dim.set_power(base_quantity, power)
        return dim
