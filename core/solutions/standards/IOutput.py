# -*- encoding: utf-8 -*-
"""
Copyright (C) 2024, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Interface for output exchange items.
"""
from __future__ import annotations
from core.solutions.standards.IBaseExchangeItem import IBaseExchangeItem
from core.solutions.standards.IValueSet import IValueSet

from abc import abstractmethod
from typing import Optional


class IOutput(IBaseExchangeItem):
    """
    An output exchange item that can deliver values from an `ILinkableComponent`.

    If an output item doesn't provide the data in the way a consumer would like
    to have, the output can be adapted by an `IAdaptedOutput`, which can
    transform the data according to the consumer's wishes.
    """

    @property
    @abstractmethod
    def consumers(self) -> Optional[list[IInput]]:
        """Returns a list of all consumers of this output.

        The consumers list is just responsible for topology checking and
        pre-preparation, not for data delivery.
        """
        pass

    @property
    @abstractmethod
    def adapters(self) -> Optional[list[IAdaptedOutput]]:
        """Returns a list of all adapted output items."""
        pass

    @abstractmethod
    def add_consumer(self, consumer: IInput):
        """Adds a consumer to the output item."""
        pass

    @abstractmethod
    def add_adapter(self, adapter: IAdaptedOutput):
        """Adds an adapted output to the output item.

        The adapter needs to add this item as an adaptee.
        """
        pass

    @abstractmethod
    def remove_consumer(self, consumer: IInput):
        """Removes a consumer from the output item."""
        pass

    @abstractmethod
    def remove_adapter(self, adapter: IAdaptedOutput):
        """Removes an adapted output from the output item.

        The adapter needs to remove this adaptee too.
        """
        pass

    @abstractmethod
    def get_values(self, querier: IBaseExchangeItem) -> IValueSet:
        """Returns the value set of the output item.

        This method must, in accordance with the specification determine:
        + if the requested time is in the future it loops update until
        that instant is reached;
        + if it's in the past it first checks the buffer and otherwise
        interpolates or extrapolates;
        + for differing units, it applies SI conversions;
        + if the call stack is re-entered, it immediately extrapolates
        to prevent deadlock;
        + single missing value's filled with `missing_data_value`, and
        only when the entire set is unavailable's an exception thrown.

        thereby safely returning valid data for any consumer at any time.
        """
        pass
