# -*- encoding: utf-8 -*-
"""
Copyright (C) 2026, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Exchange port interfaces.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from yunmeng.interfaces.solution.IDataset import (
    IElementSet,
    IValueSet,
    Quantity,
    TimeSpan,
)
from yunmeng.interfaces.types import ArrayLike

if TYPE_CHECKING:
    from yunmeng.interfaces.solution.IModel import ILinkableModel

# ---------------------------------------------------
# region IExchangeItem
# ---------------------------------------------------


class IExchangeItem(ABC):
    """A named data port with a quantity, spatial footprint and values."""

    @property
    @abstractmethod
    def id(self) -> str: ...

    @property
    @abstractmethod
    def owner(self) -> "ILinkableModel": ...

    @property
    @abstractmethod
    def quantity(self) -> Quantity: ...

    @property
    @abstractmethod
    def element_set(self) -> IElementSet: ...

    @property
    @abstractmethod
    def time_span(self) -> TimeSpan: ...

    @property
    @abstractmethod
    def values(self) -> IValueSet: ...


# ---------------------------------------------------
# region IInput
# ---------------------------------------------------


class IInput(IExchangeItem):
    """Input port that receives data from one provider."""

    @property
    @abstractmethod
    def provider(self) -> "IOutput": ...

    @provider.setter
    @abstractmethod
    def provider(self, output: "IOutput"): ...

    @property
    @abstractmethod
    def is_connected(self) -> bool: ...

    @abstractmethod
    def pull(self) -> ArrayLike:
        """Fetch the latest value from the provider."""
        ...


# ---------------------------------------------------
# region IOutput
# ---------------------------------------------------


class IOutput(IExchangeItem):
    """Output port that can feed multiple downstreams."""

    @property
    @abstractmethod
    def adapters(self) -> list["IAdapterOutput"]: ...

    @property
    @abstractmethod
    def consumers(self) -> list[IInput]: ...

    @property
    @abstractmethod
    def version(self) -> int:
        """Version of cached data."""
        ...

    @abstractmethod
    def add_adapter(self, adapter: "IAdapterOutput"): ...

    @abstractmethod
    def remove_adapter(
        self,
        adapter: "IAdapterOutput",
    ): ...

    @abstractmethod
    def clear_adapters(self): ...

    @abstractmethod
    def add_consumer(self, consumer: IInput): ...

    @abstractmethod
    def remove_consumer(
        self,
        consumer: IInput,
    ): ...

    @abstractmethod
    def clear_consumers(self): ...

    @abstractmethod
    def get_values(
        self,
        requester: IInput = None,
    ) -> ArrayLike: ...

    @abstractmethod
    def add_values(self, values: ArrayLike): ...

    @abstractmethod
    def set_values(self, values: ArrayLike): ...


# ---------------------------------------------------
# region IAdapterOutput
# ---------------------------------------------------


class IAdapterOutput(IOutput):
    """Transform data from one output layout to another
    (spatial, temporal, unit)."""

    @property
    @abstractmethod
    def adaptee(self) -> IOutput:
        """The output to adapt from."""
        ...

    @adaptee.setter
    @abstractmethod
    def adaptee(self, adaptee: IOutput): ...

    @abstractmethod
    def then(self, next_adapter: "IAdapterOutput") -> "IAdapterOutput":
        """Fluent chaining: a.then(b) wires a -> b and returns b."""
        ...

    @abstractmethod
    def adapt(self, data: ArrayLike) -> ArrayLike: ...

    @abstractmethod
    def refresh(self): ...
