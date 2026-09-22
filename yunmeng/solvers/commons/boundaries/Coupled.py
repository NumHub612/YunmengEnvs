# -*- encoding: utf-8 -*-
"""
Copyright (C) 2026, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

description
"""

from __future__ import annotations
import numpy as np

from yunmeng.interfaces.solution import IInput
from yunmeng.interfaces.solver import IBoundaryCondition
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

    def __init__(
        self,
        bc_id: str,
        target_field: str,
        region: Region,
        port: IInput,
        channel: str = "constraints",
    ):
        self._id = bc_id
        self._field = target_field
        self._region = region
        self._port = port
        self._channel = channel
        self._tag = "coupled"

    @classmethod
    def get_name(cls) -> str:
        return "CoupledBC"

    @property
    def id(self) -> str:
        return self._id

    @property
    def target_field(self) -> str:
        return self._field

    @property
    def semantic_tag(self) -> str:
        return self._tag

    @property
    def region(self) -> Region:
        return self._region

    def evaluate(self, t: float) -> tuple:
        frame = np.asarray(self._port.pull(), dtype=float).flatten()
        n = len(self._region.element_ids)
        if frame.size == 1:
            vals = np.full(n, frame[0])
        elif frame.size == n:
            vals = frame
        else:
            raise ValueError(
                f"CoupledBoundary '{self._id}': provider frame has {frame.size} "
                f"values, region has {n} elements."
            )
        return self._channel, vals
