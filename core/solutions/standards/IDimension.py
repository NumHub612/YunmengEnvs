# -*- encoding: utf-8 -*-
"""
Copyright (C) 2024, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

To provide dimensions management in physical quantities.
"""
from enum import Enum
from dataclasses import dataclass


class DimensionBase(Enum):
    """
    Base dimensions for physical quantities.
    """

    Length = 0  # Base dimension length.
    Mass = 1  # Base dimension mass.
    Time = 2  # Base dimension time.
    ElectricCurrent = 3  # Base dimension electric current.
    Temperature = 4  # Base dimension temperature.
    AmountOfSubstance = 5  # Base dimension amount of substance.
    LuminousIntensity = 6  # Base dimension luminous intensity.
    Currency = 7  # Base dimension currency.


@dataclass
class IDimension:
    """To provide dimension-related operations."""

    powers: dict[DimensionBase, float] = {d: 0 for d in DimensionBase}

    def get_power(self, base_quantity: DimensionBase) -> float:
        """
        Gets the power for the requested dimension.
        """
        return self.powers[base_quantity]

    def set_power(
        self,
        base_quantity: DimensionBase,
        power: float,
    ):
        """
        Sets a power for a base dimension.
        """
        self.powers[base_quantity] = power
