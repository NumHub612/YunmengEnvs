# -*- encoding: utf-8 -*-
"""
Copyright (C) 2024, The YunmengEnvs Contributors. Join us, for you talents!  

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

    def from_np(self, arr: np.ndarray) -> "Coordinate":
        """
        Convert from numpy array.
        """
        return Coordinate(arr[0], arr[1], arr[2])


@dataclass
class Node:
    """
    Node element for the mesh.
    """

    id: int
    coord: Coordinate


@dataclass
class Face:
    """
    Face element for the mesh.

    NOTE:
        - A face may be a point(for 1d), line(for 2d), or polygon(for 3d).
    """

    id: int
    nodes: list
    center: Coordinate
    perimeter: float
    area: float
    normal: Vector


@dataclass
class Cell:
    """
    Cell element for the mesh.
    """

    id: int
    faces: list
    center: Coordinate
    surface: float
    volume: float
