# -*- encoding: utf-8 -*-
"""
Copyright (C) 2026, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Lightweight input port implementation.
"""

import numpy as np

from yunmeng.interfaces.solution import (
    IElementSet,
    IInput,
    ILinkableModel,
    IOutput,
    Quantity,
    TimeSpan,
)
from yunmeng.interfaces.types import ArrayLike
from yunmeng.solutions.commons.dataset import FrameValueSet


def _as_frame(values: ArrayLike):
    return np.atleast_1d(np.asarray(values, dtype=float))


class BaseInput(IInput):
    """Single-provider input port."""

    def __init__(
        self,
        item_id: str,
        quantity: Quantity,
        elements: IElementSet,
        time_span: TimeSpan = None,
        owner: "ILinkableModel " = None,
        required: bool = True,
    ):
        self._id = item_id
        self._quantity = quantity
        self._elements = elements
        self._time_span = time_span or TimeSpan()
        self._owner = owner
        self._required = required
        self._provider = None
        self._values = None

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
    def values(self) -> FrameValueSet:
        if self._values is None:
            return None
        return FrameValueSet(self._quantity, self._values)

    @property
    def required(self) -> bool:
        return self._required

    @property
    def provider(self):
        return self._provider

    @provider.setter
    def provider(self, output: "IOutput "):
        self._provider = output

    @property
    def is_connected(self) -> bool:
        return self._provider is not None

    def pull(self):
        if self._provider is None:
            raise ValueError(f"Input {self._id} has no provider.")
        self._values = _as_frame(self._provider.get_values(self))
        return self._values

    def get_values(self, requester: "IInput " = None):
        return self._values

    def set_values(self, values: ArrayLike):
        self._values = _as_frame(values)
