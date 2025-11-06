# -*- encoding: utf-8 -*-
"""
Copyright (C) 2024, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Interface class for exchange items.
"""
from __future__ import annotations
from core.solutions.standards.ISpatialDefinition import ISpatialDefinition
from core.solutions.standards.IValueDefinition import IValueDefinition
from core.solutions.standards.IIdentifiable import IIdentifiable
from core.solutions.standards.IValueSet import IValueSet
from core.solutions.standards.ITimeSet import ITimeSet

from abc import abstractmethod
from typing import Optional


class IBaseExchangeItem(IIdentifiable):
    """Class presenting a item that can be exchanged, defining Where, When, and What.
    either as an input or output.

    The owning component is responsible for assigning the element set, timeset and
    value set to the exchange item, and also maintaining the consistency of the data.
    """

    @property
    @abstractmethod
    def spatial_definition(self) -> Optional[ISpatialDefinition]:
        """The spatial definition of the exchange item.

        The `ISpatialDefinition` should never be returned directly; all implementing
        classes should return either `IElementSet`, or a custom
        derived spatial definition interface.
        """
        pass

    @property
    @abstractmethod
    def time_set(self) -> Optional[ITimeSet]:
        """The time set of the exchange item."""
        pass

    @property
    @abstractmethod
    def value_definition(self) -> IValueDefinition:
        """The value definition of exchange item.

        The `IValueDefinition` should never be returned directly; all implementing
        classes should return either `IQuality`, `IQuantity`, or a custom
        derived value definition interface.
        """
        pass

    @property
    @abstractmethod
    def values(self) -> IValueSet:
        """The values of the exchange item.

        For an input item, this is an implicit `get_values()` call on the provider.
        It should be called during the `update()` method after
        the component's `prepare()` phase.

        For an output item, this is the 'original'  valueset provided,
        without any adaptation logic. For the engine or adapter to directly
        access the underlying data.
        """
        pass

    @property
    @abstractmethod
    def component(self) -> ILinkableComponent:
        """The owner component of the exchange item.

        For output exchange item, this is the component responsible for providing
        the content of the output item. It's possible for an exchange item to
        have no owner, in this case the method will return none.
        """
        pass

    @property
    @abstractmethod
    def event_manager(self) -> EventManager:
        """The event manager for this exchange item.

        This event manager must be able to register event handlers that accepting
        the `ExchangeItemChangeEventArgs` as argument.
        """
        pass
