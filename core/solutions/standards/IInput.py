# -*- encoding: utf-8 -*-
"""
Copyright (C) 2024, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Interface for input items.
"""
from core.solutions.standards.IBaseExchangeItem import IBaseExchangeItem

# from core.solutions.standards.IOutput import IOutput

from abc import abstractmethod
from typing import List


class IInput(IBaseExchangeItem):
    """
    An input item can accept values for an `ILinkableComponent`.
    """

    @abstractmethod
    def get_providers(self):
        """Gets the providers for this input item."""
        pass

    @abstractmethod
    def add_provider(self, provider):
        """Adds a provider for this input item."""
        pass

    @abstractmethod
    def remove_provider(self, provider):
        """Removes specified provider from this input item."""
        pass
