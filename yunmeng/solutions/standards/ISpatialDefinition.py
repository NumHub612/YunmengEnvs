# -*- encoding: utf-8 -*-
"""
Copyright (C) 2024, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Interface for spatial definitions.
"""
from yunmeng.solutions.standards.IDescribable import IDescribable

from dataclasses import dataclass


@dataclass
class ISpatialDefinition(IDescribable):
    """General spatial interface that all other spatial constructions extend."""

    # Specifies the OGC Well-Known Text(WKT) representation of
    # spatial reference system to be used in association
    # with the coordinates in the `ISpatialDefinition`.
    spatial_reference_system: str = ""

    # Number of data elements in the spatial axis.
    element_count: int = 0

    # The version number for the spatial axis.
    # The version must be incremented if anything inside
    # the spatial axis changed.
    version: int = 0
