# -*- encoding: utf-8 -*-
"""
Copyright (C) 2024, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Interface for input items.
"""
from __future__ import annotations
from yunmeng.solutions.standards.IBaseExchangeItem import IBaseExchangeItem
from yunmeng.solutions.standards.IOutput import IOutput

from abc import abstractmethod
from typing import Optional


class IInput(IBaseExchangeItem):
    """
    An input item can accept values for an `ILinkableComponent`.

    An input item can only have one provider.
    """

    @property
    @abstractmethod
    def provider(self) -> Optional[IOutput]:
        """Gets the provider for this input item."""
        pass

    @provider.setter
    @abstractmethod
    def provider(self, provider: IOutput):
        """Sets the provider.

        While resetting the provider, this provider should also
        add the input to the its consumer list.
        And the previsous provider should remove the input item
        from its consumer list.
        """
        pass
