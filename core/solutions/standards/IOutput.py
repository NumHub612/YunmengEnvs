# -*- encoding: utf-8 -*-
"""
Copyright (C) 2024, The YunmengEnvs Contributors. Join us, for you talents!  
URL: https://github.com/NumHub612/YunmengEnvs  
License: Apache License 2.0

Interface for output exchange items.
"""
from abc import abstractmethod
from typing import List

from core.solutions.standards.IBaseExchangeItem import IBaseExchangeItem
from core.solutions.standards.IInput import IInput


class IOutput(IBaseExchangeItem):
    """
    An output exchange item that can deliver values from an `ILinkableComponent`.

    If an output does not provide the data in the way a consumer would like to have,
    the output can be adapted by an `IAdaptedOutput`, which can transform
    the data according to the consumer's wishes.
    """

    from core.solutions.standards.IAdaptedOutput import IAdaptedOutput

    @abstractmethod
    def get_consumers(self) -> List[IInput]:
        """Returns a list of all consumers of this output."""
        pass

    @abstractmethod
    def add_consumer(self, consumer: IInput):
        """Adds a consumer to the output item. Every input item needs
        to add itself as a consumer first.
        """
        pass

    @abstractmethod
    def remove_consumer(self, consumer: IInput):
        """Removes a consumer from the output item."""
        pass

    @abstractmethod
    def get_adapters(self) -> List[IAdaptedOutput]:
        """Returns a list of all adapted outputs of this output."""
        pass

    @abstractmethod
    def remove_adapter(self, adaptedOutput: "IAdaptedOutput"):
        """Removes an adapted output from the output item."""
        pass

    @abstractmethod
    def add_adapter(self, adaptedOutput: IAdaptedOutput):
        """Adds an adapted output to the output item.

        Every adapted output item needs to add itself as an adaptee.
        """
        pass