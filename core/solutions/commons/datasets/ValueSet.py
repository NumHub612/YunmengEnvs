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
            self._values = np.array(values, dtype=value_definition.value_type)
        else:
            self._values = np.full(
                shape,
                value_definition.missing_data_value,
                dtype=value_definition.value_type,
            )

    @property
    def value_definition(self) -> IValueDefinition:
        return self._value_definition

    @property
    def shape(self) -> tuple[int]:
        # TODO: check if all values are valid.
        return self._values.shape

    def set_or_add_values(self, indices: tuple[int], values: Any):
        if len(indices) < 4:
            self._values[tuple(indices)] = values
        else:
            raise ValueError("Invalid indices.")

    def remove_values(self, indices: tuple[int]):
        if len(indices) < 4:
            self._values[tuple(indices)] = self._value_definition.missing_data_value
        else:
            raise ValueError("Invalid indices.")

    def get_values_for_element(self, element_index: int) -> list[Any]:
        return self._values[:, element_index]

    def get_values_for_time(self, time_index: list[int]) -> list[Any]:
        return self._values[time_index, :]

    def __len__(self) -> int:
        return self._values.shape[0]

    def __getittem__(self, indices: tuple[int]) -> Any:
        return self._values[tuple(indices)]

    def __setitem__(self, indices: tuple[int], value: Any):
        self._values[tuple(indices)] = value
