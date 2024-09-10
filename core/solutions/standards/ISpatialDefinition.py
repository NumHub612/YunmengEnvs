# -*- encoding: utf-8 -*-
"""
Copyright (C) 2024, The YunmengEnvs Contributors. Join us, for you talents!  
URL: https://github.com/NumHub612/YunmengEnvs  
License: Apache License 2.0

Interface for spatial definitions.
"""
from abc import abstractmethod

from core.solutions.standards.IDescribable import IDescribable


class ISpatialDefinition(IDescribable):
    """General spatial construct that all other spatial constructions extend from."""

    @property
    @abstractmethod
    def spatial_reference_system(self) -> str:
        """
        Specifies the OGC Well-Known Text(WKT) representation of
        spatial reference system to be used in association with the coordinates in
        the `ISpatialDefinition`.
        """
        pass

    @property
    @abstractmethod
    def element_count(self) -> int:
        """Number of data elements in the spatial axis."""
        pass

    @property
    @abstractmethod
    def version(self) -> int:
        """
        The version number for the spatial axis.

        The version must be incremented if anything inside the spatial axis changed,
        or if an entirely new spatial axis is provided.
        """
        pass