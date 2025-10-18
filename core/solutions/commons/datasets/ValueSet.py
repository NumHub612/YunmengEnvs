# -*- encoding: utf-8 -*-
"""
Copyright (C) 2025, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

ValueSet used to store values of a specific variable.
"""
from core.solutions.standards import IValueSet, IValueDefinition
import numpy as np
from typing import Any


class ValueSet(IValueSet):
    """ValueSet class."""

    def __init__(self, value_definition: IValueDefinition, shape: tuple[int]):
        self._value_definition = value_definition
        self._shape = [0, 0]
        self._values = np.full(
            shape,
            value_definition.missing_value,
            dtype=value_definition.value_type,
        )

    @property
    def value_definition(self) -> IValueDefinition:
        return self._value_definition

    @property
    def shape(self) -> tuple[int]:
        return tuple(self._shape)

    def set_or_add_values(self, indices: tuple[int], values: Any):
        if len(indices) < 4:
            self._values[tuple(indices)] = values

            # Update valid shape
            if indices[0] >= self._shape[0]:
                self._shape = (indices[0] + 1, *self._shape[1:])
            if len(indices) > 1 and indices[1] >= self._shape[1]:
                self._shape = (*self._shape[:1], indices[1] + 1)
        else:
            raise ValueError("Invalid indices.")

    def remove_values(self, indices: tuple[int]):
        if len(indices) < 4:
            self._values[tuple(indices)] = self._value_definition.missing_value

            # Update valid shape
            if self._shape[0] > 1 and indices[0] == self._shape[0] - 1:
                self._shape = (self._shape[0] - 1, *self._shape[1:])
            if self._shape[1] > 1 and indices[1] == self._shape[1] - 1:
                self._shape = (*self._shape[:1], self._shape[1] - 1)
        else:
            raise ValueError("Invalid indices.")

    def get_values_for_element(self, element_index: int) -> list[Any]:
        return self._values[:, element_index]

    def get_values_for_time(self, time_index: list[int]) -> list[Any]:
        return self._values[time_index, :]
