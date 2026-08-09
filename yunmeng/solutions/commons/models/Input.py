# -*- encoding: utf-8 -*-
"""
Copyright (C) 2026, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Lightweight input port implementation.
"""

from __future__ import annotations
import numpy as np

from yunmeng.solutions.standards import (
    ILinkableModel,
    IInput,
    IOutput,
    IElementSet,
    IValueSet,
    Quantity,
    TimeSpan,
)
from yunmeng.solutions.commons.datasets import FrameValueSet


class BaseInput(IInput):
    """Single-provider input port backed by a numpy frame."""

    def __init__(
        self,
        item_id: str,
        quantity: Quantity,
        elements: IElementSet,
        time_span: TimeSpan = None,
        owner: ILinkableModel = None,
        required: bool = True,
    ):
        self._id = item_id
        self._quantity = quantity
        self._elements = elements
        self._time_span = time_span or TimeSpan()
        self._owner = owner
        self._required = required
        self._provider: IOutput = None
        self._values: np.ndarray = None

    # -- IExchangeItem --------------------------------

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
        return self._elements

    @property
    def time_span(self) -> TimeSpan:
        return self._time_span

    @property
    def values(self) -> IValueSet:
        if self._values is None:
            return None
        return FrameValueSet(self._quantity, self._values)

    # -- IInput ----------------------------------------

    @property
    def required(self) -> bool:
        """Whether this input must be connected before running."""
        return self._required

    @property
    def provider(self) -> IOutput:
        return self._provider

    @provider.setter
    def provider(self, output: IOutput):
        self._provider = output

    @property
    def is_connected(self) -> bool:
        return self._provider is not None

    def pull(self) -> np.ndarray:
        if self._provider is None:
            raise ValueError(f"Input {self._id} has no provider.")
        self._values = _as_frame(self._provider.get_values(self))
        return self._values

    # -- direct frame access ----------------------------

    def get_values(self, requester: IInput = None) -> np.ndarray:
        """Values last pulled (an input never re-queries here)."""
        return self._values

    def set_values(self, values: np.ndarray):
        self._values = _as_frame(values)


def _as_frame(values) -> np.ndarray:
    return np.atleast_1d(np.asarray(values, dtype=float))
