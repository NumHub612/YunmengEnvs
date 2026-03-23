# -*- encoding: utf-8 -*-
"""
Copyright (C) 2024, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Node, face, and cell classes for the mesh.
"""
from dataclasses import dataclass
import numpy as np


@dataclass(slots=True)
class Coordinate:
    """
    Coordinate.
    """

    x: float = 0
    y: float = 0
    z: float = 0

    @staticmethod
    def from_numpy(arr: np.ndarray) -> "Coordinate":
        return Coordinate(arr[0], arr[1], arr[2])

    def to_numpy(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z])

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


@dataclass(slots=True)
class Element:
    """
    Element base class.
    """

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

    nodes: list[int]


@dataclass(slots=True)
class Cell(Element):
    """
    Cell element.
    """

    faces: list[int]
