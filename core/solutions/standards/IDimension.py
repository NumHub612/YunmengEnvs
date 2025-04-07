# -*- encoding: utf-8 -*-
"""
Copyright (C) 2024, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Interface for dimensions in physical quantities.
"""
from enum import Enum
from abc import ABC, abstractmethod


class DimensionBase(Enum):
    Length = 0  # Base dimension length.
    Mass = 1  # Base dimension mass.
    Time = 2  # Base dimension time.
    ElectricCurrent = 3  # Base dimension electric current.
    Temperature = 4  # Base dimension temperature.
    AmountOfSubstance = 5  # Base dimension amount of substance.
    LuminousIntensity = 6  # Base dimension luminous intensity.
    Currency = 7  # Base dimension currency.


class IDimension(ABC):
    """Interface for dimension-related operations."""

    @abstractmethod
    def get_power(self, base_quantity: DimensionBase) -> float:
        """
        Gets the power for the requested dimension.
        """
        pass

    @abstractmethod
    def set_power(self, base_quantity: DimensionBase, power: float):
        """
        Sets a power for a base dimension.
        """
        pass
