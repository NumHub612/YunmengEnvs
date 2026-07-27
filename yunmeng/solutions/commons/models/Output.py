# -*- encoding: utf-8 -*-
"""
Copyright (C) 2026, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Lightweight output port implementation.
"""

from __future__ import annotations
from typing import Any, Optional
import numpy as np

from yunmeng.solutions.standards import (
    IOutput,
    IInput,
    IElementSet,
    IValueSet,
    IExchangeAdapter,
    Quantity,
    TimeSpan,
)


class SimpleValueSet(IValueSet):
    """Minimal value set wrapping a numpy array."""

    def __init__(self, quantity: Quantity, values: np.ndarray):
        self._quantity = quantity
        self._values = np.asarray(values)

    @property
    def quantity(self) -> Quantity:
        return self._quantity

    @property
    def values(self) -> np.ndarray:
        return self._values

    def set_values(self, values: np.ndarray):
        self._values = np.asarray(values)


class BaseOutput(IOutput):
    """Simple output port backed by a numpy array."""

    def __init__(
        self,
        item_id: str,
        quantity: Quantity,
        element_set: IElementSet,
        time_span: TimeSpan = None,
        producer: callable = None,
    ):
        self._id = item_id
        self._quantity = quantity
        self._element_set = element_set
        self._time_span = time_span or TimeSpan()
        self._producer = producer
        self._consumers: list[IInput] = []
        self._adapters: list[IExchangeAdapter] = []
        self._cache: Optional[np.ndarray] = None
        self._component: Any = None
        self._generation: int = 0

    @property
    def component(self) -> Any:
        return self._component

    @property
    def version(self) -> int:
        return self._generation

    @property
    def id(self) -> str:
        return self._id

    @property
    def quantity(self) -> Quantity:
        return self._quantity

    @property
    def element_set(self) -> IElementSet:
        return self._element_set

    @property
    def time_span(self) -> TimeSpan:
        return self._time_span

    @property
    def values(self) -> IValueSet:
        return SimpleValueSet(self._quantity, self._cache)

    @property
    def consumers(self) -> list[IInput]:
        return self._consumers

    def add_consumer(self, consumer: IInput):
        if consumer not in self._consumers:
            self._consumers.append(consumer)

    def remove_consumer(self, consumer: IInput):
        if consumer in self._consumers:
            self._consumers.remove(consumer)

    def add_adapter(self, adapter: IExchangeAdapter):
        if adapter not in self._adapters:
            self._adapters.append(adapter)

    def remove_adapter(self, adapter_id: str):
        self._adapters = [a for a in self._adapters if a.id != adapter_id]

    def set_cache(self, values: np.ndarray):
        self._cache = np.asarray(values)
        self._generation += 1

    def get_values(self, requester: Optional[IInput] = None) -> np.ndarray:
        if self._producer is not None and self._cache is None:
            self.set_cache(self._producer())
        if self._cache is None:
            raise ValueError(f"Output {self._id} has no data.")
        data = self._cache
        for adapter in self._adapters:
            if adapter.can_adapt(self, requester):
                data = adapter.adapt(data, self, requester)
        return data

    def __repr__(self) -> str:
        return f"BaseOutput({self._id}, consumers={len(self._consumers)})"
