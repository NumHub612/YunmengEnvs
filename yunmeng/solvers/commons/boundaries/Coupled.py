# -*- encoding: utf-8 -*-
"""
Copyright (C) 2026, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Coupled boundary condition.
"""

from __future__ import annotations
import numpy as np

from yunmeng.interfaces.solution import IInput
from yunmeng.interfaces.solver import IBoundaryCondition, BoundaryChannel
from yunmeng.interfaces.supports import Region
from yunmeng.interfaces.types import ArrayLike
from yunmeng.numerics.algos import ym_register


@ym_register("boundary")
class CoupledBoundary(IBoundaryCondition):
    """Port-to-BC bridge: an IBoundaryCondition whose value is pulled from
    a coupled input port each time the solver evaluates its boundary
    provider (at t+dt of every step).

    This keeps the coupling lazy and time-aligned: the alternative of
    rebuilding a constant BC in the model's update() would freeze the
    value for the whole step and duplicate the provider's assembly logic.
    """

    @classmethod
    def get_name(cls) -> str:
        return "CoupledBC"

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

    def __init__(self, bc_id: str, target_field: str, region: Region, port: IInput):
        if not region or not region.element_ids:
            raise ValueError(f"BC {bc_id} requires a valid region.")

        self._id = bc_id
        self._field = target_field
        self._region = region
        self._port = port

    def evaluate(self, t: float) -> ArrayLike:
        frame = np.asarray(self._port.pull(), dtype=float).flatten()
        n = len(self._region.element_ids)
        if frame.size == 1:
            vals = np.full(n, frame[0])
        elif frame.size == n:
            vals = frame
        else:
            raise ValueError(
                f"CoupledBoundary {self._id}: provider frame has {frame.size} "
                f"values, region has {n} elements."
            )
        return vals
