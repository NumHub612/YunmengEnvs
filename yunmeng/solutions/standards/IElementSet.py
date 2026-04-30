# -*- encoding: utf-8 -*-
"""
Copyright (C) 2024, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Data exchange between components is nearly always related to one or more elements
in a space, either geo-referenced or not. An elementset can be a list of 2D or 3D
spatial elements or as a special case, a list of ID based (non spatial) elements.
Possible element types are defined in `ElementType`.

For 3D elements (i.e. polyhedron) the shape can be queried by face. When the
elementset is geo-referenced, coordinates (X, Y, Z) can be obtained for each
node of an element.

A geo-referenced elementset needs to have a valid 'SpatialReferenceSystem'
property set in a `ISpatialDefinition`. This is a string that specifies the OGC
WKT representation of a spatial reference. An empty string indicates that there
in no spatial reference, which is only valid if the `ElementType` is `IdBased`.

While an `IElementSet` can be used to query the geometric description of a
model schematization, it does not necessarily provide all topological knowledge
on inter-element connections.
"""
from __future__ import annotations
from yunmeng.solutions.standards.ISpatialDefinition import ISpatialDefinition

from abc import abstractmethod


class IElementSet(ISpatialDefinition):
    """An list of elements having a common type."""

    @property
    @abstractmethod
    def element_geom_type(self) -> GeomType:
        """The geometry type of elements."""
        pass

    @property
    @abstractmethod
    def element_count(self) -> int:
        """The number of elements."""
        pass

    @abstractmethod
    def get_element_index(self, element_id: str) -> int:
        """Returns the index of the element, or `None` if not found."""
        pass

    @abstractmethod
    def get_element_id(self, element_index: int) -> str:
        """Returns the ID of the element, or `None` if not found."""
        pass

    @abstractmethod
    def get_face_count(self, element_index: int) -> int:
        """Returns the number of faces, or `None` if not polyhedron."""
        pass

    @abstractmethod
    def get_node_count(self, element_index: int) -> int:
        """Returns the number of nodes, or `None` if it's ID based."""
        pass

    @abstractmethod
    def get_face_node_indices(
        self,
        element_index: int,
        face_index: int,
    ) -> list[int]:
        """Returns the nodes indices of the face."""
        pass

    @abstractmethod
    def get_node_coordinates(
        self,
        element_index: int,
        node_index: int,
    ) -> list[float]:
        """Returns the 3d coordinates of the node."""
        pass
