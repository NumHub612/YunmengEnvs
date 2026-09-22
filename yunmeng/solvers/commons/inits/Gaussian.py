# -*- encoding: utf-8 -*-
"""
Copyright (C) 2026, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

description
"""

import numpy as np
from yunmeng.interfaces.types import ArrayLike
from yunmeng.interfaces.supports import IField
from yunmeng.interfaces.solver import IInitialCondition
from yunmeng.numerics.algos import ym_register


@ym_register("init")
class GaussianIC(IInitialCondition):
    """Gaussian initial pulse on cell centers."""

    def __init__(
        self,
        ic_id: str,
        target_field: str,
        centers: ArrayLike,
        center: float = 0.5,
        width: float = 0.08,
        amplitude: float = 1.0,
    ):
        if width <= 0:
            raise ValueError("GaussianIC width must be positive")
        self._id = ic_id
        self._field = target_field
        self._centers = np.asarray(centers, dtype="float64")
        self._center = float(center)
        self._width = float(width)
        self._amplitude = float(amplitude)

    @classmethod
    def get_name(cls) -> str:
        return "GaussianIC"

    @property
    def id(self) -> str:
        return self._id

    @property
    def target_field(self) -> str:
        return self._field

    def get(self, **kwargs) -> ArrayLike:
        distance = self._centers - self._center
        return self._amplitude * np.exp(-(distance**2) / (2.0 * self._width**2))

    def apply(self, field: IField):
        field.values = self.get()
