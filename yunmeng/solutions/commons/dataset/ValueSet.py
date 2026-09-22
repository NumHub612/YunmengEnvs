# -*- encoding: utf-8 -*-
"""
Copyright (C) 2025, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

ValueSet used to store values of a specific variable.
"""

import numpy as np

from yunmeng.interfaces.solution import IValueSet, Quantity
from yunmeng.interfaces.types import ArrayLike


class FrameValueSet(IValueSet):
    """IValueSet view over a single (elements, components) frame."""

    def __init__(self, quantity: Quantity, frame: ArrayLike):
        self._quantity = quantity
        self._frame = np.atleast_2d(np.asarray(frame, dtype=float))

    @property
    def quantity(self) -> Quantity:
        return self._quantity

    @property
    def shape(self) -> tuple:
        return (1, *self._frame.shape)

    def get_values(self, time_idx: int = -1):
        return self._frame

    def set_values(self, time_idx: int, values: ArrayLike):
        self._frame = np.atleast_2d(np.asarray(values, dtype=float))

    def append_values(self, values: ArrayLike):
        self.set_values(0, values)
