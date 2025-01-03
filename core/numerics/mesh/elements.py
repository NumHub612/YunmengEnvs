# -*- encoding: utf-8 -*-
"""
Copyright (C) 2024, The YunmengEnvs Contributors. Join us, share your ideas!  

Node, face, and cell classes for the mesh.
"""
from core.numerics.fields import Vector
from dataclasses import dataclass
import numpy as np


@dataclass
class Coordinate:
    """
    Coordinate.
    """

    x: float = 0
    y: float = 0
    z: float = 0

    def __add__(self, other: "Coordinate") -> "Coordinate":
        """
        Add two coordinates.
        """
        return Coordinate(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: "Coordinate") -> "Coordinate":
        """
        Subtract two coordinates.
        """
        return Coordinate(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, other: float) -> "Coordinate":
        """
        Multiply a coordinate by a scalar.
        """
        return Coordinate(self.x * other, self.y * other, self.z * other)

    def __rmul__(self, other: float) -> "Coordinate":
        """
        Multiply a scalar by a coordinate.
        """
        return self.__mul__(other)

    def __truediv__(self, other: float) -> "Coordinate":
        """
        Divide a coordinate by a scalar.
        """
        return Coordinate(self.x / other, self.y / other, self.z / other)

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


@dataclass
class Element:
    """
    Element base class.
    """

    id: int
    coordinate: Coordinate


@dataclass
class Node(Element):
    """
    Node element for the mesh.
    """

    pass


@dataclass
class Face(Element):
    """
    Face element for the mesh.
    """

    nodes: list
    perimeter: float
    area: float
    normal: Vector


@dataclass
class Cell(Element):
    """
    Cell element for the mesh.
    """

    faces: list
    surface: float
    volume: float
