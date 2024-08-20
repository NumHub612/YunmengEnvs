# -*- encoding: utf-8 -*-
"""
Copyright (C) 2024, The YunMengEnvs Project Contributors. Join us !
License: Apache License 2.0

Interfaces for dimensions in physical quantities.
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
    @abstractmethod
    def get_power(self, base_quantity: DimensionBase) -> float:
        """
        Gets the power for the requested dimension.

        Examples:
            For a quantity such as flow, which may have the unit m3/s,
            the `get_power` method must work as follows:
        >>>
            Flow.get_power(DimensionBase.AmountOfSubstance) -->  0
            Flow.get_power(DimensionBase.Currency)          -->  0
            Flow.get_power(DimensionBase.ElectricCurrent)   -->  0
            Flow.get_power(DimensionBase.Length)            -->  3
            Flow.get_power(DimensionBase.LuminousIntensity) -->  0
            Flow.get_power(DimensionBase.Mass)              -->  0
            Flow.get_power(DimensionBase.Temperature)       -->  0
            Flow.get_power(DimensionBase.Time)              --> -1
        """
        pass

    @abstractmethod
    def set_power(self, base_quantity: DimensionBase, power: float):
        """
        Sets a power for a base dimension.

        Parameters:
            base_quantity: The base dimension.
            power: The power.
        """
        pass
