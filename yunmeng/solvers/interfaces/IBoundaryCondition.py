# -*- encoding: utf-8 -*-
"""
Copyright (C) 2024, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Interfaces for boundary conditions at faces of a mesh.
"""
from abc import ABC, abstractmethod
from typing import Dict, Optional, Union, Any
from enum import Enum, auto
from dataclasses import dataclass
from yunmeng.numerics.fields.variables import Variable


@dataclass
class BoundaryValue:
    """Boundary condition value container."""

    # Prescribed value (water level, velocity, temperature, etc.)
    value: Optional[Variable] = None

    # Prescribed flux (discharge, mass flux, momentum flux, etc.)
    flux: Optional[Variable] = None

    # Type-specific extra data (coefficients, etc.)
    extra: Optional[Dict[str, Any]] = None


class BoundaryType(Enum):
    """Environmental fluid mechanics boundary types."""

    # Fundamental mathematical types
    VALUE = auto()
    FLUX = auto()
    MIXED = auto()

    # Environmental fluid specifics
    OPEN = auto()
    WALL = auto()
    COUPLED = auto()


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
        The instance name.
        """
        pass

    @abstractmethod
    def update(self):
        """
        Updates the boundary value.
        """
        pass

    @abstractmethod
    def evaluate(self) -> BoundaryValue:
        """
        Gets current boundary condition.
        """
        pass
