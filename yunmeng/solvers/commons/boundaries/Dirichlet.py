# -*- encoding: utf-8 -*-
"""
Copyright (C) 2026, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

description
"""

from __future__ import annotations
from typing import Callable
import numpy as np

from yunmeng.interfaces.solver import IBoundaryCondition
from yunmeng.interfaces.supports import Region
from yunmeng.interfaces.types import ArrayLike


class DirichletBC(IBoundaryCondition):
    """Constant or time-dependent prescribed value on a mesh region."""

    def __init__(
        self,
        bc_id: str,
        target_field: str,
        region: Region,
        value: float | Callable[[float], float],
        tag: str = "open",
    ):
        self._id = bc_id
        self._field = target_field
        self._region = region
        self._value = value
        self._tag = tag

    @classmethod
    def get_name(cls) -> str:
        return "DirichletBC"

    @property
    def id(self) -> str:
        return self._id

    @property
    def semantic_tag(self) -> str:
        return self._tag

    @property
    def target_field(self) -> str:
        return self._field

    @property
    def region(self) -> Region:
        return self._region

    def evaluate(self, t: float) -> tuple[str, ArrayLike]:
        value = self._value(t) if callable(self._value) else self._value
        return "constraints", np.full(len(self._region.element_ids), float(value))
