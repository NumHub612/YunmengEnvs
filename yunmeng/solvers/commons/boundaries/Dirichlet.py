# -*- encoding: utf-8 -*-
"""
Copyright (C) 2026, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Dirichlet boundary condition specifed the boundary value directly.
"""

from __future__ import annotations
import numpy as np

from yunmeng.standard.solver import IBoundaryCondition, BoundaryChannel
from yunmeng.standard.supports import Region
from yunmeng.standard.types import ArrayLike

from yunmeng.numerics.algos import ym_register
from yunmeng.numerics.fields import Timeseries


@ym_register("boundary")
class DirichletBC(IBoundaryCondition):
    """Constant or time-dependent prescribed value on a mesh region."""

    @classmethod
    def get_name(cls) -> str:
        return "DirichletBC"

    @property
    def id(self) -> str:
        return self._id

    @property
    def channel(self) -> BoundaryChannel:
        return BoundaryChannel.CONSTRAINTS

    @property
    def target_field(self) -> str:
        return self._field

    @property
    def region(self) -> Region:
        return self._region

    def __init__(
        self,
        bc_id: str,
        target_field: str,
        region: Region,
        value: float | ArrayLike | Timeseries,
    ):
        if region is None or region.element_ids.shape[0] == 0:
            raise ValueError(f"BC {bc_id} requires a valid region.")

        self._id = bc_id
        self._field = target_field
        self._region = region
        self._value = value
        self._n_elements = len(region.element_ids)

    def evaluate(self, t: float) -> ArrayLike:
        if isinstance(self._value, Timeseries):
            value = np.full(self._n_elements, self._value(t))
        elif isinstance(self._value, (int, float)):
            value = np.full(self._n_elements, self._value)
        else:
            value = self._value
        return value
