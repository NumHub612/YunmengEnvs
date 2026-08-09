# -*- encoding: utf-8 -*-
"""
Copyright (C) 2026, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Lightweight output port implementation.
"""

from __future__ import annotations
import numpy as np

from yunmeng.solutions.standards import (
    ILinkableModel,
    IOutput,
    IInput,
    IAdapterOutput,
    IElementSet,
    IValueSet,
    Quantity,
    TimeSpan,
)
from yunmeng.solutions.commons.datasets import FrameValueSet


class BaseOutput(IOutput):
    """Output port holding the current data frame."""

    def __init__(
        self,
        item_id: str,
        quantity: Quantity,
        elements: IElementSet,
        time_span: TimeSpan = None,
        owner: ILinkableModel = None,
    ):
        self._id = item_id
        self._quantity = quantity
        self._element_set = elements
        self._time_span = time_span or TimeSpan()
        self._owner = owner
        self._consumers: list[IInput] = []
        self._adapters: list[IAdapterOutput] = []
        self._cache: np.ndarray = None
        self._generation: int = 0

    # -- IExchangeItem -----------------------------------

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
    def values(self) -> IValueSet:
        if self._cache is None:
            return None
        return FrameValueSet(self._quantity, self._cache)

    # -- IOutput ------------------------------------------

    @property
    def adapters(self) -> list:
        return self._adapters

    @property
    def consumers(self) -> list[IInput]:
        return self._consumers

    @property
    def version(self) -> int:
        return self._generation

    def add_adapter(self, adapter):
        if adapter not in self._adapters:
            adapter.adaptee = self
            self._adapters.append(adapter)

    def remove_adapter(self, adapter):
        if adapter in self._adapters:
            adapter.adaptee = None
            self._adapters.remove(adapter)

    def clear_adapters(self):
        for adapter in self._adapters:
            adapter.adaptee = None
        self._adapters.clear()

    def add_consumer(self, consumer: IInput):
        if consumer not in self._consumers:
            self._consumers.append(consumer)
            consumer.provider = self

    def remove_consumer(self, consumer: IInput):
        if consumer in self._consumers:
            self._consumers.remove(consumer)
            consumer.provider = None

    def clear_consumers(self):
        for consumer in self._consumers:
            consumer.provider = None
        self._consumers.clear()

    # -- data ----------------------------------------------

    def _publish(self, values: np.ndarray):
        # bumps version, refreshes adapters
        self._cache = _as_frame(values)
        self._generation += 1
        for adapter in list(self._adapters):
            adapter.refresh()

    def add_values(self, values: np.ndarray):
        """Publish a new frame."""
        self._publish(values)

    def set_values(self, values: np.ndarray):
        """Overwrite the current frame."""
        self._publish(values)

    def get_values(self, requester: IInput = None) -> np.ndarray:
        if self._cache is None:
            raise ValueError(f"Output {self._id} has no data.")
        return self._cache

    # -- snapshot support ------------------------------------

    def _state(self) -> tuple:
        if self._cache is not None:
            c = self._cache.copy()
        else:
            c = None
        return (c, self._generation)

    def _set_state(self, state: tuple):
        cache, generation = state
        self._cache = None if cache is None else cache.copy()
        self._generation = generation
        for adapter in list(self._adapters):
            adapter.refresh()


def _as_frame(values) -> np.ndarray:
    return np.atleast_1d(np.asarray(values, dtype=float))
