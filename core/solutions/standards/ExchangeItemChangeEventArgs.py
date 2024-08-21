# -*- encoding: utf-8 -*-
"""
Copyright (C) 2024, The YunmengEnvs Contributors. Join us, for you talents!  
URL: https://github.com/NumHub612/YunmengEnvs  
License: Apache License 2.0

Interface provides the args for an ExchangeItemValueChanged event.
"""
from abc import ABC, abstractmethod
from typing import Optional

from core.solutions.standards.IBaseExchangeItem import IBaseExchangeItem


class ExchangeItemChangeEventArgs(ABC):
    """
    To provides the information that will be passed when
    firing an `ExchangeItemValueChanged` event.
    """

    @property
    @abstractmethod
    def exchange_item(self) -> Optional[IBaseExchangeItem]:
        """The exchange item that has been changed."""
        pass

    @exchange_item.setter
    @abstractmethod
    def exchange_item(self, obj: IBaseExchangeItem):
        """Sets the exchange item that has been changed."""
        pass

    @property
    @abstractmethod
    def message(self) -> str:
        """The message description of the change."""
        pass

    @message.setter
    @abstractmethod
    def message(self, value: str):
        """Sets the message description of the change."""
        pass
