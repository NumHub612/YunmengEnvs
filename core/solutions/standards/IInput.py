# -*- encoding: utf-8 -*-
"""
Copyright (C) 2024, The YunMengEnvs Project Contributors. Join us !
License: Apache License 2.0

Interfaces for input items.
"""
from abc import abstractmethod
from typing import List, Optional

from .IBaseExchangeItem import IBaseExchangeItem
from .IOutput import IOutput


class IInput(IBaseExchangeItem):
    """
    An input item can accept values for an `ILinkableComponent`.
    """

    @abstractmethod
    def get_providers(self) -> List[Optional[IOutput]]:
        """Gets the providers for this input item."""
        pass

    @abstractmethod
    def add_provider(self, provider: IOutput) -> None:
        """Adds a provider for this input item."""
        pass

    @abstractmethod
    def remove_provider(self, provider: IOutput) -> None:
        """Removes specified provider from this input item."""
        pass
