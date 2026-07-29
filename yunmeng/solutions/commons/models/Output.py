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
from yunmeng.solutions.commons.datasets import SimpleValueSet


class BaseOutput(IOutput):
    """Simple output port backed by a numpy array."""

    def __init__(
        self,
        item_id: str,
        quantity: Quantity,
        elements: IElementSet,
        time_span: TimeSpan = None,
        model: ILinkableModel = None,
        producer: IOutput = None,
    ):
        self._id = item_id
        self._quantity = quantity
        self._element_set = elements
        self._time_span = time_span or TimeSpan()
        self._producer: IOutput = producer
        self._consumers: list[IInput] = []
        self._adapters: list[IAdapterOutput] = []
        self._cache: np.ndarray = None
        self._component: ILinkableModel = model
        self._generation: int = 0

    # -- instance properties ------------------------

    @property
    def adapters(self) -> list[IAdapterOutput]:
        return self._adapters

    @property
    def consumers(self) -> list[IInput]:
        return self._consumers

    @property
    def model(self) -> ILinkableModel:
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

    # -- instance methods ---------------------------

    def add_adapter(self, adapter: IAdapterOutput):
        if adapter not in self._adapters:
            adapter.adaptee = self
            self._adapters.append(adapter)

    def remove_adapter(self, adapter: IAdapterOutput):
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

    def add_values(self, values: np.ndarray):
        self._cache = np.asarray(values)
        self._generation += 1
        for adapter in self._adapters:
            adapter.refresh()

    def add_values(self, values: np.ndarray):
        self._cache = np.asarray(values)
        for adapter in self._adapters:
            adapter.refresh()

    def get_values(self, requester: IInput = None) -> np.ndarray:
        if self._cache is None and self._producer is not None:
            self.add_values(self._producer.get_values(requester))
        if self._cache is None:
            raise ValueError(f"Output {self._id} has no data.")
        return self._cache
