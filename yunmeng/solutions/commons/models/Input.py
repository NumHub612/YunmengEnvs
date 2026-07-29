# -*- encoding: utf-8 -*-
"""
Copyright (C) 2026, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Lightweight input port implementation.
"""

from __future__ import annotations
import numpy as np

from yunmeng.solutions.standards import (
    IInput,
    IOutput,
    IElementSet,
    IValueSet,
    Quantity,
    TimeSpan,
)


class BaseInput(IInput):
    """Simple input port backed by a numpy array."""

    def __init__(
        self,
        item_id: str,
        quantity: Quantity,
        elements: IElementSet,
        time_span: TimeSpan = None,
    ):
        self._id = item_id
        self._quantity = quantity
        self._elements = elements
        self._time_span = time_span or TimeSpan()
        self._provider: IOutput = None
        self._values: np.ndarray = None

    # -- instance properties ------------------------

    @property
    def id(self) -> str:
        return self._id

    @property
    def quantity(self) -> Quantity:
        return self._quantity

    @property
    def element_set(self) -> IElementSet:
        return self._elements

    @property
    def time_span(self) -> TimeSpan:
        return self._time_span

    @property
    def values(self) -> IValueSet:
        return None

    @property
    def provider(self) -> IOutput:
        return self._provider

    @provider.setter
    def provider(self, output: IOutput):
        if self._provider is not None:
            self._provider.remove_consumer(self)
        self._provider = output
        if output is not None:
            output.add_consumer(self)

    @property
    def is_connected(self) -> bool:
        return self._provider is not None

    # -- instance methods ---------------------------

    def pull(self) -> np.ndarray:
        if self._provider is None:
            raise ValueError(f"Input {self._id} has no provider.")
        self._values = self._provider.get_values(self)
        return self._values

    def get_values(self, requester: IInput = None) -> np.ndarray:
        """Values last pulled into this input.
        + None before the first pull;
        + an input does not re-query its provider here."""
        return self._values

    def set_values(self, values: np.ndarray):
        """Set values directly."""
        self._values = values
