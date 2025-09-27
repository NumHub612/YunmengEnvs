# -*- encoding: utf-8 -*-
"""
Copyright (C) 2024, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Interface provides the args for an ExchangeItemValueChanged event.
"""
from core.solutions.standards.IBaseExchangeItem import IBaseExchangeItem

from abc import ABC, abstractmethod


class ExchangeItemChangeEventArgs(ABC):
    """
    To provides the information that will be passed when
    firing an `ExchangeItemValueChanged` event.
    """

    @property
    @abstractmethod
    def exchange_item(self) -> IBaseExchangeItem:
        """The exchange item that has been changed."""
        pass

    @exchange_item.setter
    @abstractmethod
    def exchange_item(self, obj: IBaseExchangeItem):
        """Sets the exchange item."""
        pass

    @property
    @abstractmethod
    def message(self) -> str:
        """The message description of the change."""
        pass

    @message.setter
    @abstractmethod
    def message(self, value: str):
        """Sets the message description."""
        pass
