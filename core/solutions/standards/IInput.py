# -*- encoding: utf-8 -*-
"""
Copyright (C) 2024, The YunmengEnvs Contributors. Join us, for you talents!  
URL: https://github.com/NumHub612/YunmengEnvs  
License: Apache License 2.0

Interface for input items.
"""
from abc import abstractmethod
from typing import List

from core.solutions.standards.IBaseExchangeItem import IBaseExchangeItem
from core.solutions.standards.IOutput import IOutput


class IInput(IBaseExchangeItem):
    """
    An input item can accept values for an `ILinkableComponent`.
    """

    @abstractmethod
    def get_providers(self) -> List[IOutput]:
        """Gets the providers for this input item."""
        pass

    @abstractmethod
    def add_provider(self, provider: IOutput):
        """Adds a provider for this input item."""
        pass

    @abstractmethod
    def remove_provider(self, provider: IOutput):
        """Removes specified provider from this input item."""
        pass
