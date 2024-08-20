# -*- encoding: utf-8 -*-
"""
Copyright (C) 2024, The YunMengEnvs Project Contributors. Join us !
License: Apache License 2.0

Interfaces for quantity values.
"""
from abc import abstractmethod
from typing import Optional

from .IUnit import IUnit
from .IValueDefinition import IValueDefinition


class IQuantity(IValueDefinition):
    """
    Class specifies values as an amount of some unit.
    """

    @abstractmethod
    def get_unit(self) -> Optional[IUnit]:
        """Unit of quantity."""
        pass
