# -*- encoding: utf-8 -*-
"""
Copyright (C) 2024, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Node, face, and cell classes for the mesh.
"""
from dataclasses import dataclass
import numpy as np
import enum


class MeshDim(enum.Enum):
    """The mesh dimensions."""

    DIM1 = "1d"
    DIM2 = "2d"
    DIM3 = "3d"
    NONE = "none"

    @staticmethod
    def from_str(dim: str) -> "MeshDim":
        """Convert a string to a MeshDim."""
        if dim == "1d":
            return MeshDim.DIM1
        elif dim == "2d":
            return MeshDim.DIM2
        elif dim == "3d":
            return MeshDim.DIM3
        else:
            return MeshDim.NONE


class ElementType(enum.Enum):
    """Element types in CFD."""

    CELL = "cell"
    FACE = "face"
    # EDGE = "edge"
    NODE = "node"
    NONE = "none"  # might be useful for id-based elements

    @staticmethod
    def from_str(etype: str) -> "ElementType":
        """Convert a string to an ElementType."""
        if etype == "cell":
            return ElementType.CELL
        elif etype == "face":
            return ElementType.FACE
        elif etype == "edge":
            return ElementType.EDGE
        elif etype == "node":
            return ElementType.NODE
        else:
            return ElementType.NONE


@dataclass(slots=True)
class Coordinate:
    """
    Coordinate.
    """

    x: float = 0
    y: float = 0
    z: float = 0

    def to_np(self) -> np.ndarray:
        """
        Convert to numpy array.
        """
        return np.array([self.x, self.y, self.z])

    @staticmethod
    def from_np(arr: np.ndarray) -> "Coordinate":
        """
        Convert from numpy array.
        """
        return Coordinate(arr[0], arr[1], arr[2])

    def __add__(self, other: "Coordinate") -> "Coordinate":
        return Coordinate(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: "Coordinate") -> "Coordinate":
        return Coordinate(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, other: float) -> "Coordinate":
        return Coordinate(self.x * other, self.y * other, self.z * other)

    def __rmul__(self, other: float) -> "Coordinate":
        return self.__mul__(other)

    def __truediv__(self, other: float) -> "Coordinate":
        return Coordinate(self.x / other, self.y / other, self.z / other)

    def __eq__(self, other: "Coordinate", tolerance: float = 1e-6):
        return (
            abs(self.x - other.x) < tolerance
            and abs(self.y - other.y) < tolerance
            and abs(self.z - other.z) < tolerance
        )

    def __ne__(self, other: "Coordinate", tolerance: float = 1e-6):
        return not self.__eq__(other, tolerance)


@dataclass(slots=True)
class Element:
    """
    Element base class.
    """

    # Element id contiguously from 0 or uncontiguous
    id: int
    coordinate: Coordinate


@dataclass(slots=True)
class Node(Element):
    """
    Node element.
    """

    pass


@dataclass(slots=True)
class Face(Element):
    """
    Face element.
    """

    # Sorted list of node indices
    nodes: list[int]


@dataclass(slots=True)
class Cell(Element):
    """
    Cell element.
    """

    # List of face indices
    faces: list[int]
