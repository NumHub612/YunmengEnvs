# -*- encoding: utf-8 -*-
"""
Copyright (C) 2026, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Baseic boundary condition class.
"""

from yunmeng.solvers.interfaces import IBoundaryCondition, BoundaryType, BoundaryValue
from yunmeng.numerics.mesh import Mesh, Region
from yunmeng.setting import logger

import numpy as np


class BaseBoundary(IBoundaryCondition):
    """
    Abstract base class for boundary conditions.
    """

    def __init__(
        self,
        id: str,
        target_field: str,
        region: Region,
    ):
        """
        Args:
            id: Unique identifier for this BC instance.
            target_field: Name of the field this BC applies to (e.g., "u").
            region: Region defining the boundary location.
        """
        self._id = id
        self._target_field = target_field
        self._bc: BoundaryValue = None
        self._region = region

        self._resolved_ids = region.get_element_ids()

    @property
    def target_field(self) -> str:
        return self._target_field

    @target_field.setter
    def target_field(self, new_field: str):
        self._target_field = new_field

    @property
    def region(self) -> Region:
        return self._region

    @region.setter
    def region(self, new_region: Region):
        self._region = new_region
        self._resolved_ids = None

    @property
    def id(self) -> str:
        return self._id

    def get(self) -> BoundaryValue:
        # Default assuming it's a uniform constant value.
        return self._bc
