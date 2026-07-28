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
    """Input port that receives data from one provider output."""

    @property
    @abstractmethod
    def provider(self) -> Optional[IOutput]:
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
    """Output port that can feed multiple downstream inputs."""

    @property
    @abstractmethod
    def consumers(self) -> list[IInput]:
        pass

    @property
    @abstractmethod
    def component(self) -> Any:
        """Model that owns this output port."""
        pass

    @property
    @abstractmethod
    def version(self) -> int:
        """Version of the cached data."""
        pass

    @abstractmethod
    def add_consumer(self, consumer: IInput):
        pass

    @abstractmethod
    def remove_consumer(self, consumer: IInput):
        pass

    @abstractmethod
    def add_adapter(self, adapter: IExchangeAdapter):
        pass

    @abstractmethod
    def remove_adapter(self, adapter_id: str):
        pass

    @abstractmethod
    def get_values(
        self,
        requester: Optional[IInput] = None,
    ) -> np.ndarray:
        """Return current data payload."""
        pass


# ---------------------------------------------------
# region IAdapter
# ---------------------------------------------------


class IExchangeAdapter(ABC):
    """Transform data from one output layout to another
    (spatial, temporal, unit)."""

    @property
    @abstractmethod
    def id(self) -> str:
        pass

    @abstractmethod
    def adapt(
        self,
        data: np.ndarray,
        source: IOutput,
        target: IInput,
    ) -> np.ndarray:
        pass

    @classmethod
    @abstractmethod
    def can_adapt(
        cls,
        source: IOutput,
        target: IInput,
    ) -> bool:
        pass
