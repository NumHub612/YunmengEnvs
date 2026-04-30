# -*- encoding: utf-8 -*-
"""
Copyright (C) 2025, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

ValueSet used to store values of a specific variable.
"""
from yunmeng.solutions.standards import IValueSet, IValueDefinition, IQuantity
import numpy as np
from typing import Any
from copy import deepcopy


class ValueSet(IValueSet):
    """ValueSet class supporting float values only."""

    def __init__(
        self,
        value_definition: IValueDefinition,
        shape: tuple[int],
        values: np.ndarray = None,
    ):
        self._value_definition = value_definition
        self._shape = shape
        if values is not None:
            if values.shape != shape:
                raise ValueError("Invalid shape of values.")
            # TODO: check if all values are valid.
            self._values = np.array(values)
        else:
            self._values = np.full(
                shape,
                value_definition.missing_data_value,
            )

    @property
    def value_definition(self) -> IValueDefinition:
        return self._value_definition

    @property
    def shape(self) -> tuple[int]:
        "(n_times, n_elements, n_variables)."
        return self._values.shape

    def set_or_add_values(self, indices: tuple[int], values: Any):
        # TODO: check if all values are valid.
        if indices[0] >= self._shape[0]:
            # add values for new time
            values = np.array(values).reshape(1, -1)
            self._values = np.append(self._values, values, axis=0)
        elif len(indices) == 2 and indices[1] >= self._shape[1]:
            # add values for new element
            values = np.array(values).reshape(1, -1)
            self._values = np.append(self._values, values, axis=1)
        else:
            self._values[tuple(indices)] = values

    def remove_values(self, indices: tuple[int]):
        if len(indices) == 1:
            # remove values for a time
            self._values = np.delete(self._values, indices[0], axis=0)
        elif len(indices) == 2:
            # remove values for an element
            self._values = np.delete(self._values, indices[1], axis=1)
        else:
            self._values[tuple(indices)] = self._value_definition.missing_data_value

    def get_values_for_element(self, element_index: int) -> np.ndarray:
        values = self._values[:, element_index]
        new_data = []
        for v in values:
            data = deepcopy(self._value_definition)
            data.value = v
            new_data.append(data)
        new_data = np.array(new_data).reshape(1, -1)
        return new_data

    def get_values_for_time(self, time_index: list[int]) -> np.ndarray:
        values = self._values[time_index, :]
        new_data = []
        for v in values:
            data = deepcopy(self._value_definition)
            data.value = v
            new_data.append(data)
        new_data = np.array(new_data).reshape(1, -1)
        return new_data

    def __len__(self) -> int:
        return self._values.shape[0]

    def __getitem__(self, indices: tuple[int]) -> IQuantity | np.ndarray:
        if len(indices) == 1:
            # get values for a time
            values = self._values[indices[0], :]
            new_data = []
            for v in values:
                data = deepcopy(self._value_definition)
                data.value = v
                new_data.append(data)
            new_data = np.array(new_data)
            return new_data
        else:
            # get values for an element at a time
            data = deepcopy(self._value_definition)
            data.value = self._values[indices]
            return data

    def __setitem__(self, indices: tuple[int], value: Any):
        self._values[tuple(indices)] = value
