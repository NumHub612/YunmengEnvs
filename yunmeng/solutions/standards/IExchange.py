# -*- encoding: utf-8 -*-
"""
Copyright (C) 2026, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Exchange item interfaces — IExchangeItem, IInput, IOutput, IExchangeAdapter.

An exchange item describes WHAT data flows (IQuantity), WHERE it lives
(IElementSet), WHEN it is valid (ITimeSpan), and holds the actual numeric
payload (IValueSet).
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional
import numpy as np

from IData import IElementSet, IQuantity, ITimeSpan, IValueSet

# ---------------------------------------------------
# region IExchangeItem
# ---------------------------------------------------


class IExchangeItem(ABC):
    """Base descriptor for a data item that can be exchanged between
    components.Carries the full metadata needed for spatial/temporal
    interpolation, unit conversion, and type checking."""

    @property
    @abstractmethod
    def id(self) -> str:
        """Unique identifier."""
        pass

    @property
    @abstractmethod
    def quantity(self) -> IQuantity:
        """Quantity definition (What)."""
        pass

    @property
    @abstractmethod
    def element_set(self) -> IElementSet:
        """Spatial definition (Where)."""
        pass

    @property
    @abstractmethod
    def time_span(self) -> ITimeSpan:
        """Temporal definition (When)."""
        pass

    @property
    @abstractmethod
    def values(self) -> IValueSet:
        """Data payload."""
        pass


# ---------------------------------------------------
# region IInput
# ---------------------------------------------------


class IInput(IExchangeItem):
    """Input port of a kinkable component. Receives data from at most
    one provider (IOutput) via PULL or LOOP coupling."""

    @property
    @abstractmethod
    def provider(self) -> Optional[IOutput]:
        """The upstream output that feeds this input."""
        pass

    @provider.setter
    @abstractmethod
    def provider(self, output: IOutput):
        pass

    @property
    @abstractmethod
    def is_connected(self) -> bool:
        """Whether a provider has been assigned."""
        pass

    @abstractmethod
    def pull(self) -> np.ndarray:
        """Fetch the latest value from the provider.

        Called by the Scheduler (PULL mode)or by the IterativeCoupler
        (LOOP mode) to move data across the link.
        """
        pass


# ---------------------------------------------------
# region IOutput
# ---------------------------------------------------


class IOutput(IExchangeItem):
    """Output port of linkable component. May be consumed by multiple
    downstream inputs and optionally transformed by IExchangeAdapters."""

    @property
    @abstractmethod
    def consumers(self) -> list[IInput]:
        """All downstream inputs connected to this output."""
        pass

    @abstractmethod
    def remove_consumer(self, consumer: IInput):
        """Remove a consumer."""
        pass

    @abstractmethod
    def add_consumer(self, consumer: IInput):
        """Register a consumer."""
        pass

    @abstractmethod
    def get_values(self, requester: Optional[IInput] = None) -> np.ndarray:
        """Return the current data payload.

        If adapters are registered and *requester* is provided, the
        adapters may transform the data to match the requester's
        spatial/temporal/unit requirements.
        """
        pass

    @abstractmethod
    def add_adapter(self, adapter: IExchangeAdapter):
        """Attach data-transform adapter (e.g. spatial interpolator,
        unit converter, temporal averager)."""
        pass

    @abstractmethod
    def remove_adapter(self, adapter_id: str):
        """Detach an adapter by its id."""
        pass


# ---------------------------------------------------
# region IExchangeAdapter
# ---------------------------------------------------


class IExchangeAdapter(ABC):
    """Transform data flowing from a source exchange item to meet the
    requirements of a target exchange item.

    An adapter sits on an IOutput and transforms data before reaching
    downstream consumers.  Typical uses:
      - Spatial interpolation   : source grid → target grid / points
      - Temporal interpolation  : source step → target step
      - Unit conversion         : m3/s → mm/h, etc.
      - Aggregation             : per-cell → basin-average
    """

    @property
    @abstractmethod
    def id(self) -> str:
        pass

    @abstractmethod
    def adapt(
        self, data: np.ndarray, source: IExchangeItem, target: IExchangeItem
    ) -> np.ndarray:
        """Transform *data* from *source* layout to *target* layout."""
        pass

    @classmethod
    @abstractmethod
    def can_adapt(cls, source: IExchangeItem, target: IExchangeItem) -> bool:
        """Return whether this adapter is capable of transforming
        from *source* to *target*."""
        pass
