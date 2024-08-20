# -*- encoding: utf-8 -*-
"""
Copyright (C) 2024, The YunMengEnvs Project Contributors. Join us !
License: Apache License 2.0

Interfaced class for element sets.
"""
from enum import Enum
from abc import abstractmethod
from typing import List

from .ISpatialDefinition import ISpatialDefinition


class ElementType(Enum):
    """Shape Type of elements in an `IElementSet`."""

    IdBased = 0
    Point = 1
    Polyline = 2
    Polygon = 3
    Polyhedron = 4


class IElementSet(ISpatialDefinition):
    """An list of elements having a common type."""

    @abstractmethod
    def get_element_type(self) -> ElementType:
        """Returns the type of elements in this set."""
        pass

    @abstractmethod
    def get_element_index(self, element_id: str) -> int:
        """Returns the index of the element with the given ID."""
        pass

    @abstractmethod
    def get_element_id(self, index: int) -> str:
        """Returns the ID of the element at the given index."""
        pass

    @abstractmethod
    def get_face_count(self, element_index: int) -> int:
        """Returns the number of faces."""
        pass

    @abstractmethod
    def get_node_count(self, element_index: int) -> int:
        """Returns the number of nodes."""
        pass

    @abstractmethod
    def get_face_node_indices(self, element_index: int, face_index: int) -> List[int]:
        """Returns the indices of the nodes that make up the face."""
        pass

    @abstractmethod
    def get_x(self, element_index: int, node_index: int) -> float:
        """Returns the x-coordinate of the node."""
        pass

    @abstractmethod
    def get_y(self, element_index: int, node_index: int) -> float:
        """Returns the y-coordinate of the node."""
        pass

    @abstractmethod
    def get_z(self, element_index: int, node_index: int) -> float:
        """Returns the z-coordinate of the node."""
        pass

    @abstractmethod
    def get_element_count(self) -> int:
        """Returns the number of elements in this set."""
        pass
