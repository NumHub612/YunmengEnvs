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

    x: float
    y: float
    z: float

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

    def to_np(self) -> np.ndarray:
        """
        Convert to numpy array.
        """
        return np.array([self.x, self.y, self.z])


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
