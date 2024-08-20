# -*- encoding: utf-8 -*-
"""
Copyright (C) 2024, The YunMengEnvs Project Contributors. Join us !
License: Apache License 2.0

Interfaced class for exchange items.
"""
from abc import ABC, abstractmethod
from typing import Callable, Optional, Any

from .IValueDefinition import IValueDefinition
from .IValueSet import IValueSet
from .IElementSet import IElementSet
from .ITimeSet import ITimeSet
from .ILinkableComponent import ILinkableComponent


class IBaseExchangeItem(ABC):
    """Class presenting a item that can be exchanged, either as an input or output."""

    @abstractmethod
    def get_value_definition(self) -> Optional[IValueDefinition]:
        """Definition of the values in the exchange item."""
        pass

    @abstractmethod
    def reset(self):
        """Resets the exchange item."""
        pass

    @abstractmethod
    def get_values(self) -> Optional[IValueSet]:
        """Provides the values matching the value definition specified."""
        pass

    @abstractmethod
    def get_elementset(self) -> Optional[IElementSet]:
        """Gets the exchange item's elements."""
        pass

    @abstractmethod
    def set_values(self, value: IValueSet):
        """Sets the exchange item's values."""
        pass

    @abstractmethod
    def get_timeset(self) -> Optional[ITimeSet]:
        """Gets the exchange item's available time set."""
        pass

    @abstractmethod
    def set_timeset(self, times: ITimeSet):
        """Sets the exchange item's time set."""
        pass

    @abstractmethod
    def set_elementset(self, elements: IElementSet):
        """Sets the exchange item's elements."""
        pass

    @abstractmethod
    def get_component(self) -> Optional[ILinkableComponent]:
        """Gets the owner of the exchange item."""
        pass

    @abstractmethod
    def add_listener(self, func: Callable):
        """Adds a listener to the exchange item."""
        pass

    @abstractmethod
    def remove_listener(self, func: Callable):
        """Removes the listener."""
        pass
