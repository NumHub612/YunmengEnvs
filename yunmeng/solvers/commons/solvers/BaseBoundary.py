# -*- encoding: utf-8 -*-
"""
Copyright (C) 2026, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Baseic boundary condition class.
"""

# -*- encoding: utf-8 -*-
"""
Copyright (C) 2025, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

To provide the Dirichlet boundary condition.
"""

from yunmeng.solvers.interfaces import IBoundaryCondition, BoundaryType, BoundaryValue
from yunmeng.numerics.mesh import Mesh, Region, get_element_ids

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

        self._mesh: Mesh = None
        self._resolved_ids: np.ndarray = None

    @property
    def target_field(self) -> str:
        return self._target_field

    @property
    def region(self) -> Region:
        return self._region

    @property
    def id(self) -> str:
        return self._id

    def attach(self, mesh: Mesh):
        self._mesh = mesh

        if self._region is None:
            raise ValueError(f"BC {self._id}: region is not set")

        self._resolved_ids = get_element_ids(mesh, self._region)

    def validate(self):
        if self._target_field is None:
            raise ValueError(f"BC {self._id}: target_field is not set")
        if self._mesh is None:
            raise ValueError(f"BC {self._id}: mesh is not attached")
        if self._resolved_ids is None or len(self._resolved_ids) == 0:
            raise ValueError(f"BC {self._id}: no boundary elements resolved")

    def get(self) -> BoundaryValue:
        # Default assuming it's a uniform constant value.
        return self._bc

    def reset(self) -> None:
        self._mesh = None
        self._resolved_ids = None
