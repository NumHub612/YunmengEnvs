# -*- encoding: utf-8 -*-
"""
Copyright (C) 2024, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Interface for valueset.
"""
from core.solutions.standards.IValueDefinition import IValueDefinition

from abc import ABC, abstractmethod
from typing import Any


class IValueSet(ABC):
    """Class represents a general(ordered) multi-dimensional set of values,
    according to the `ITimeSet` and `IElementSet`.

    The values should be stored in the structure: 'timestamp*element*value'.

    In more general scenario,
    the size of each dimension can vary, depending on the indices provided, e.g.
    in a 2D matrix, each row can have different lengths. For example,
    assuming the data is stored as a double[][] matrix, then matrix[1].size
    need not equal to matrix[2].size.
    """

    @property
    @abstractmethod
    def value_definition(self) -> IValueDefinition:
        """
        Definition of the values in the value set.
        """
        pass

    @property
    @abstractmethod
    def shape(self) -> tuple[int]:
        """
        Current shape of the value set.
        """
        pass

    @abstractmethod
    def set_or_add_values(self, indices: tuple[int], values: Any):
        """
        Sets or adds the value object specified by the given indices.
        """
        pass

    @abstractmethod
    def remove_values(self, indices: tuple[int]):
        """
        Removes the values specified by the given indices.

        It is possible to remove not just a single value, but also
        the whole set of values for the given indices.
        """
        pass

    @abstractmethod
    def get_values_for_element(self, element_index: int) -> list[Any]:
        """
        Gets the values, for all times, for the given element index.
        If the data is spatial independent, element_index
        must be specified as 0.
        """
        pass

    @abstractmethod
    def get_values_for_time(self, time_index: list[int]) -> list[Any]:
        """
        Gets the values, for all elements, for the given time index.
        If the data is temporal independent, time_index
        must be specified as 0.
        """
        pass
