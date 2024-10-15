# -*- encoding: utf-8 -*-
"""
Copyright (C) 2024, The YunmengEnvs Contributors. Join us, for you talents!  
 
Interface for Value Set.
"""
from core.solutions.standards.IValueDefinition import IValueDefinition

from abc import ABC, abstractmethod
from typing import Any, List


class IValueSet(ABC):
    """Class represents a general(ordered) multi-dimensional set of values."""

    @abstractmethod
    def get_number_of_indices(self) -> int:
        """
        Returns the number of possible indices (dimensions).

        Returns:
            number of indices, zero based.
        """
        pass

    @abstractmethod
    def get_index_count(self, indices: List[int]) -> int:
        """
        Returns the length of the dimension specified.

        To get the size of the specified dimension, use zero-length int array as input
        argument. Length of indice must be a least one
        smaller than the `get_number_of_indices()`.

        Parameters:
            indices: Indices of the dimension to get the size of.

        Returns:
            Length of the specified dimension.
        """
        pass

    @abstractmethod
    def get_value(self, indices: List[int]) -> Any:
        """
        Returns the value object specified by the given indices array.

        Parameters:
            indices: Indices of each dimension.

        Returns:
            The value for the given indices.
        """
        pass

    @abstractmethod
    def remove_value(self, indices: List[int]):
        """
        Removes the values specified by the given indices.

        It is possible to remove not just a single value item, but also the
        whole set of values for the given indices.

        Parameters:
            indices: Indices of specified dimension.
        """
        pass

    @abstractmethod
    def set_or_add_value(self, indices: List[int], value: Any):
        """
        Sets or adds the value object specified by the given indices.

        Parameters:
            indices: Indices of each dimension.
            value: Value object to be set or added.
        """
        pass

    @abstractmethod
    def get_values_for_element(self, element_index: int) -> List[Any]:
        """
        Gets the values, for all times, for the given elementIndex.
        If the data is spatial independent, element_index must be specified as 0.

        Parameters:
            element_index: Index of element in `IElementSet`.

        Returns:
            The timeseries values.
        """
        pass

    @property
    @abstractmethod
    def value_definition(self) -> IValueDefinition:
        """
        Definition of the values in the value set.
        """
        pass
