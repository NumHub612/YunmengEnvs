# -*- encoding: utf-8 -*-
"""
Copyright (C) 2025, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

ValueSet used to store values of a specific variable.
"""

from yunmeng.solutions.standards import IValueSet, Quantity
import numpy as np
from typing import Any
from copy import deepcopy

# ---------------------------------------------------
# region FrameValueSet
# ---------------------------------------------------


class FrameValueSet(IValueSet):
    """IValueSet view over a single (elements, components) frame.

    This's the minimal container-semantics adapter for IValueSet:
    ports keep raw frames for speed; consumers asking for
    `port.values` receive this view.
    """

    def __init__(self, quantity: Quantity, frame: np.ndarray):
        self._quantity = quantity
        self._frame = np.atleast_2d(np.asarray(frame, dtype=float))

    @property
    def quantity(self) -> Quantity:
        return self._quantity

    @property
    def shape(self) -> tuple:
        return (1, *self._frame.shape)

    def get_values(self, time_idx: int = -1) -> np.ndarray:
        return self._frame

    def set_values(self, time_idx: int, values: np.ndarray):
        self._frame = np.atleast_2d(np.asarray(values, dtype=float))

    def append_values(self, values: np.ndarray):
        self.set_values(0, values)
