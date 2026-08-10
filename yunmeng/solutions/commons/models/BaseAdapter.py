# -*- encoding: utf-8 -*-
"""
Copyright (C) 2026, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Base implementation of `IAdapterOutput`: adapters as output decorators.

An adapter is an output port: it wraps an upstream Output, pulls from
it, transforms the values, and serves the result to its own consumers.
"""

from __future__ import annotations
import numpy as np

from yunmeng.solutions.standards import (
    IAdapterOutput,
    IOutput,
    IInput,
    Quantity,
    IElementSet,
    TimeSpan,
)


class BaseAdapter(IAdapterOutput):
    """Base class for adapters implemented as IOutput decorators."""

    def __init__(
        self,
        adapter_id: str,
        adaptee: IOutput = None,
        quantity: Quantity = None,
        elements: IElementSet = None,
        cache: bool = True,
    ):
        self._id = adapter_id
        self._upstream = adaptee
        self._quantity = quantity
        self._element_set = elements
        self._adapters: list[IAdapterOutput] = []
        self._consumers: list[IInput] = []
        self._cache_enabled = cache
        self._cache: np.ndarray = None
        self._cache_version: int = -1

    # -- IExchangeItem ------------------------------

    @property
    def id(self) -> str:
        return self._id

    @property
    def owner(self):
        """Owner of the ultimate source port."""
        return getattr(self._upstream, "owner", None)

    @property
    def quantity(self) -> Quantity:
        if self._quantity is not None:
            return self._quantity
        if self._upstream is None:
            raise ValueError(f"Adapter '{self._id}' has no quantity.")
        return self._upstream.quantity

    @property
    def element_set(self) -> IElementSet:
        if self._element_set is not None:
            return self._element_set
        if self._upstream is None:
            raise ValueError(f"Adapter '{self._id}' has no element set.")
        return self._upstream.element_set

    @property
    def time_span(self) -> TimeSpan:
        if self._upstream is None:
            raise ValueError(f"Adapter '{self._id}' has no upstream.")
        return self._upstream.time_span

    @property
    def values(self):
        frame = self._cache
        if frame is None:
            return None
        from yunmeng.solutions.commons.datasets import FrameValueSet

        return FrameValueSet(self.quantity, frame)

    # -- IAdapterOutput -----------------------------

    @property
    def adaptee(self) -> IOutput:
        return self._upstream

    @adaptee.setter
    def adaptee(self, adaptee: IOutput):
        self._upstream = adaptee
        self._cache_version = -1

    # -- IOutput ------------------------------------

    @property
    def adapters(self) -> list[IAdapterOutput]:
        return self._adapters

    @property
    def consumers(self) -> list[IInput]:
        return self._consumers

    @property
    def model(self):
        return self.owner

    @property
    def version(self) -> int:
        return self._upstream.version if self._upstream is not None else -1

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
        self._cache = np.atleast_1d(np.asarray(values, dtype=float))
        self._cache_version = self.version
        self.refresh()

    def set_values(self, values: np.ndarray):
        self.add_values(values)

    def get_values(self, requester: IInput = None) -> np.ndarray:
        if self._upstream is None:
            raise ValueError(f"Adapter '{self._id}' has no upstream.")
        v = self._upstream.version
        if self._cache_enabled and self._cache is not None and self._cache_version == v:
            return self._cache
        data = self._upstream.get_values(self)
        out = np.atleast_1d(np.asarray(self.adapt(data), dtype=float))
        if self._cache_enabled:
            self._cache = out
            self._cache_version = v
        return out

    # -- chaining -----------------------------------

    def then(self, next_adapter: IAdapterOutput) -> IAdapterOutput:
        """``a.then(b)`` wires a -> b and returns b."""
        self.add_adapter(next_adapter)
        return next_adapter

    def refresh(self):
        self._cache_version = -1
        for adapter in self._adapters:
            adapter.refresh()

    def adapt(self, data: np.ndarray) -> np.ndarray:
        raise NotImplementedError


def chain(source: IOutput, *stages: IAdapterOutput) -> IAdapterOutput:
    """Compose ``source -> stages[0] -> ... -> stages[-1]`` and return
    the chain head (the stage to connect to the target input)."""
    prev = source
    for stage in stages:
        if stage.adaptee is None:
            stage.adaptee = prev
        prev = stage
    return prev
