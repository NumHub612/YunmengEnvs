# -*- encoding: utf-8 -*-
"""
Copyright (C) 2024, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Interfaces for boundary conditions at faces of a mesh.
"""
from abc import ABC, abstractmethod
import enum


class BoundaryType(enum.Enum):
    """Boundary type"""

    FIXED = "dirichlet"
    NATURAL = "neumann"
    MIXED = "robin"
    UNKNOWN = "unknown"


class IBoundaryCondition(ABC):
    """
    Interface for pde boundary conditions classes.
    """

    @classmethod
    @abstractmethod
    def get_name(cls) -> str:
        """
        The unique name of the boundary condition.
        """
        pass

    @classmethod
    @abstractmethod
    def get_type(cls) -> BoundaryType:
        """
        THe boundary type.
        """
        pass

    @property
    @abstractmethod
    def id(self) -> str:
        """
        The boundary condition id.
        """
        pass

    @abstractmethod
    def update(self):
        """
        Update the boundary condition parameters.
        """
        pass

    @abstractmethod
    def evaluate(self) -> tuple:
        """
        Evaluate the boundary condition.
        """
        pass
