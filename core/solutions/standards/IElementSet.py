# -*- encoding: utf-8 -*-
"""
Copyright (C) 2024, The YunmengEnvs Contributors. Join us, for you talents!  
 
Interface class for element sets.
"""
from core.solutions.standards.ISpatialDefinition import ISpatialDefinition

from enum import Enum
from abc import abstractmethod
from typing import List


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

    @property
    @abstractmethod
    def element_type(self) -> ElementType:
        """The type of elements in this set."""
        pass

    @property
    @abstractmethod
    def element_count(self) -> int:
        """The number of elements in this set."""
        pass
