# -*- encoding: utf-8 -*-
"""
Copyright (C) 2024, The YunmengEnvs Contributors. Join us, for you talents!  
 
Interface class for exchange items.
"""
from core.solutions.standards.ILinkableComponent import ILinkableComponent
from core.solutions.standards.IValueDefinition import IValueDefinition
from core.solutions.standards.IValueSet import IValueSet
from core.solutions.standards.IElementSet import IElementSet
from core.solutions.standards.ITimeSet import ITimeSet

from abc import ABC, abstractmethod
from typing import Callable, Optional


class IBaseExchangeItem(ABC):
    """Class presenting a item that can be exchanged,
    either as an input or output."""

    @abstractmethod
    def update(self) -> Optional[IValueSet]:
        """Updates the exchange item."""
        pass

    @abstractmethod
    def reset(self):
        """Resets the exchange item."""
        pass

    @property
    @abstractmethod
    def value_definition(self) -> IValueDefinition:
        """Definition of the values in the exchange item."""
        pass

    @property
    @abstractmethod
    def values(self) -> IValueSet:
        """The values."""
        pass

    @values.setter
    @abstractmethod
    def values(self, value: IValueSet):
        """Sets the exchange item's values."""
        pass

    @property
    @abstractmethod
    def elementset(self) -> IElementSet:
        """The exchange item's elements."""
        pass

    @elementset.setter
    @abstractmethod
    def elementset(self, elements: IElementSet):
        """Sets the exchange item's elements."""
        pass

    @property
    @abstractmethod
    def timeset(self) -> ITimeSet:
        """The exchange item's time set."""
        pass

    @timeset.setter
    @abstractmethod
    def timeset(self, times: ITimeSet):
        """Sets the exchange item's time set."""
        pass

    @property
    @abstractmethod
    def component(self) -> ILinkableComponent:
        """The owner of the exchange item."""
        pass

    @abstractmethod
    def add_listener(self, func: Callable):
        """Adds a listener to the exchange item."""
        pass

    @abstractmethod
    def remove_listener(self, func: Callable):
        """Removes the listener."""
        pass
