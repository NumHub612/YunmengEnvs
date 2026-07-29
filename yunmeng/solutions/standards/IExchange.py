# -*- encoding: utf-8 -*-
"""
Copyright (C) 2026, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Lightweight exchange item interfaces.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Optional
import numpy as np

from yunmeng.solutions.standards.IData import IElementSet, IValueSet, Quantity, TimeSpan

# ---------------------------------------------------
# region IExchangeItem
# ---------------------------------------------------


class IExchangeItem(ABC):
    """A named data port with a quantity, spatial footprint and values."""

    @property
    @abstractmethod
    def id(self) -> str:
        pass

    @property
    @abstractmethod
    def quantity(self) -> Quantity:
        pass

    @property
    @abstractmethod
    def element_set(self) -> IElementSet:
        pass

    @property
    @abstractmethod
    def time_span(self) -> TimeSpan:
        pass

    @property
    @abstractmethod
    def values(self) -> IValueSet:
        pass


# ---------------------------------------------------
# region IInput
# ---------------------------------------------------


class IInput(IExchangeItem):
    """Input port that receives data from one provider."""

    @property
    @abstractmethod
    def provider(self) -> IOutput:
        pass

    @provider.setter
    @abstractmethod
    def provider(self, output: IOutput):
        pass

    @property
    @abstractmethod
    def is_connected(self) -> bool:
        pass

    @abstractmethod
    def pull(self) -> np.ndarray:
        """Fetch the latest value from the provider."""
        pass


# ---------------------------------------------------
# region IOutput
# ---------------------------------------------------


class IOutput(IExchangeItem):
    """Output port that can feed multiple downstreams."""

    @property
    @abstractmethod
    def adapters(self) -> list[IAdapterOutput]:
        """All downstream adapters."""
        pass

    @property
    @abstractmethod
    def consumers(self) -> list[IInput]:
        """All downstream consumers."""
        pass

    @property
    @abstractmethod
    def model(self) -> Any:
        """Model owning this output."""
        pass

    @property
    @abstractmethod
    def version(self) -> int:
        """Version of cached data."""
        pass

    @abstractmethod
    def add_adapter(self, adapter: IAdapterOutput):
        """Add a downstream adapter."""
        pass

    @abstractmethod
    def remove_adapter(
        self,
        adapter: IAdapterOutput,
    ):
        pass

    @abstractmethod
    def clear_adapters(self):
        pass

    @abstractmethod
    def add_consumer(self, consumer: IInput):
        """Add a downstream consumer."""
        pass

    @abstractmethod
    def remove_consumer(
        self,
        consumer: IInput,
    ):
        pass

    @abstractmethod
    def clear_consumers(self):
        pass

    @abstractmethod
    def get_values(
        self,
        requester: IInput = None,
    ) -> np.ndarray:
        pass

    @abstractmethod
    def add_values(self, values: np.ndarray):
        pass


# ---------------------------------------------------
# region IAdapter
# ---------------------------------------------------


class IAdapterOutput(IOutput):
    """Transform data from one output layout to another
    (spatial, temporal, unit)."""

    @property
    @abstractmethod
    def adaptee(self) -> IOutput:
        """The output to adapt from."""
        pass

    @adaptee.setter
    @abstractmethod
    def adaptee(self, adaptee: IOutput):
        pass

    @abstractmethod
    def then(self, next_adapter: "IAdapterOutput") -> "IAdapterOutput":
        """Fluent chaining:
        `a.then(b)` wires a -> b and returns b."""
        pass

    @abstractmethod
    def adapt(
        self,
        data: np.ndarray,
    ) -> np.ndarray:
        """Transform data."""
        pass

    @abstractmethod
    def refresh(self):
        """Refresh the adapter chain."""
        pass
