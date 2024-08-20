# -*- encoding: utf-8 -*-
"""
Copyright (C) 2024, The YunMengEnvs Project Contributors. Join us !
License: Apache License 2.0

Interfaces Provides the args for an ExchangeItemValueChanged event.
"""
from abc import ABC, abstractmethod
from typing import Optional

from .IBaseExchangeItem import IBaseExchangeItem


class ExchangeItemChangeEventArgs(ABC):
    """
    To provides the information that will be passed when firing an
    `ExchangeItemValueChanged` event.
    """

    @abstractmethod
    def get_exchange_item(self) -> Optional[IBaseExchangeItem]:
        """Gets the exchange item that has been changed."""
        pass

    @abstractmethod
    def set_exchange_item(self, obj: IBaseExchangeItem):
        """Sets the exchange item that has been changed."""
        pass

    @abstractmethod
    def get_messages(self) -> str:
        """Gets the messages that will be sent to the callbacks."""
        pass

    @abstractmethod
    def set_messages(self, value: str):
        """Sets the messages that will be sent to the callbacks."""
        pass
