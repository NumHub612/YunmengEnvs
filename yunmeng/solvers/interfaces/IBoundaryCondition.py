# -*- encoding: utf-8 -*-
"""
Copyright (C) 2024, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Interfaces for boundary conditions at faces of a mesh.
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional, Union, Any
from enum import Enum, auto
from dataclasses import dataclass
from yunmeng.numerics.fields import Variable


@dataclass
class BoundaryValue:
    """Boundary condition value container."""

    value: Optional[Variable] = None  # Prescribed value
    flux: Optional[Variable] = None  # Prescribed flux
    extra: Optional[Dict[str, Any]] = None  # Type-specific extra data


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

    # @abstractmethod
    # def add_elements(self, eids: list[int]):
    #     """
    #     Adds elements to the boundary condition.
    #     """
    #     pass

    # @abstractmethod
    # def remove_elements(self, eids: list[int]):
    #     """
    #     Removes elements from the boundary condition.
    #     """
    #     pass

    @abstractmethod
    def evaluate(self, **kwargs) -> BoundaryValue:
        """
        Gets current boundary condition.
        """
        pass
