# -*- encoding: utf-8 -*-
"""
Copyright (C) 2026, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Lightweight output port implementation.
"""

import numpy as np

from yunmeng.interfaces.solution import (
    IAdapterOutput,
    IElementSet,
    IOutput,
    IInput,
    ILinkableModel,
    Quantity,
    TimeSpan,
)
from yunmeng.interfaces.types import ArrayLike
from yunmeng.solutions.commons.dataset import FrameValueSet


def _as_frame(values: ArrayLike):
    return np.atleast_1d(np.asarray(values, dtype=float))


class BaseOutput(IOutput):
    """Output port holding the current data frame."""

    def __init__(
        self,
        item_id: str,
        quantity: Quantity,
        elements: IElementSet,
        time_span: TimeSpan = None,
        owner: "ILinkableModel " = None,
    ):
        self._id = item_id
        self._quantity = quantity
        self._element_set = elements
        self._time_span = time_span or TimeSpan()
        self._owner = owner
        self._consumers: list = []
        self._adapters: list = []
        self._cache = None
        self._generation = 0

    @property
    def id(self) -> str:
        return self._id

    @property
    def owner(self):
        return self._owner

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
    def values(self) -> FrameValueSet:
        if self._cache is None:
            return None
        return FrameValueSet(self._quantity, self._cache)

    @property
    def adapters(self) -> list:
        return self._adapters

    @property
    def consumers(self) -> list:
        return self._consumers

    @property
    def version(self) -> int:
        return self._generation

    def add_adapter(self, adapter: "IAdapterOutput"):
        if adapter not in self._adapters:
            adapter.adaptee = self
            self._adapters.append(adapter)

    def remove_adapter(self, adapter: "IAdapterOutput"):
        if adapter in self._adapters:
            adapter.adaptee = None
            self._adapters.remove(adapter)

    def clear_adapters(self):
        for adapter in self._adapters:
            adapter.adaptee = None
        self._adapters.clear()

    def add_consumer(self, consumer: "IInput"):
        if consumer not in self._consumers:
            self._consumers.append(consumer)
            consumer.provider = self

    def remove_consumer(self, consumer: "IInput"):
        if consumer in self._consumers:
            self._consumers.remove(consumer)
            consumer.provider = None

    def clear_consumers(self):
        for consumer in self._consumers:
            consumer.provider = None
        self._consumers.clear()

    def _publish(self, values: ArrayLike):
        self._cache = _as_frame(values)
        self._generation += 1
        for adapter in list(self._adapters):
            adapter.refresh()

    def add_values(self, values: ArrayLike):
        self._publish(values)

    def set_values(self, values: ArrayLike):
        self._publish(values)

    def get_values(self, requester: "IInput " = None):
        if self._cache is None:
            raise ValueError(f"Output {self._id} has no data.")
        return self._cache

    # -- snapshot support ---------------------------

    def _state(self) -> tuple:
        c = None if self._cache is None else self._cache.copy()
        return (c, self._generation)

    def _set_state(self, state: tuple):
        cache, generation = state
        self._cache = None if cache is None else cache.copy()
        self._generation = generation
        for adapter in list(self._adapters):
            adapter.refresh()
