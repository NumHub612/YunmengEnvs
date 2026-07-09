# -*- encoding: utf-8 -*-
"""
Copyright (C) 2024, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Interfaces for boundary conditions at faces of a mesh.
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional, Union, Any
from enum import Enum, auto
from dataclasses import dataclass

from yunmeng.numerics.mesh import Region, Mesh
from yunmeng.numerics.fields import Variable, FieldMeta, Field


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
    def target_field(self) -> str:
        """
        The target field.
        """
        pass

    @property
    @abstractmethod
    def region(self) -> Region:
        """
        The boundary region.
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
    def attach(self, mesh: Mesh):
        """
        Attaches the mesh.
        """
        pass

    @abstractmethod
    def validate(self, **kwargs):
        """
        Validates.
        """
        pass

    @abstractmethod
    def get(self, **kwargs) -> BoundaryValue:
        """
        Gets current boundary conditions.
        """
        pass

    @abstractmethod
    def apply(self, field: Field, **kwargs):
        """
        Applys boundary condition.
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """
        Resets the boundary condition.
        """
        pass
