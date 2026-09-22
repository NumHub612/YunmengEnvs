# -*- encoding: utf-8 -*-
"""
Copyright (C) 2025, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Time-, curve- and pattern-series data structures.
"""

import numpy as np
from typing import Hashable, Any

from yunmeng.interfaces.types import ArrayLike

# ---------------------------------------------------
# region Timeseries
# ---------------------------------------------------


class Timeseries:
    """Piecewise-linear 1D series over a time axis (seconds)."""

    def __init__(self, ts_id: str, xs: ArrayLike, ys: ArrayLike):
        self._id = ts_id
        self._xs = np.asarray(xs, dtype=float).flatten()
        self._ys = np.asarray(ys, dtype=float).flatten()
        if self._xs.size != self._ys.size:
            raise ValueError(f"Timeseries '{ts_id}': xs/ys length mismatch.")
        if self._xs.size == 0:
            raise ValueError(f"Timeseries '{ts_id}': empty series.")

    @property
    def id(self) -> str:
        return self._id

    @property
    def xs(self):
        return self._xs

    @property
    def ys(self):
        return self._ys

    def value(self, t: float) -> float:
        """Linear interpolation, clamped outside the range."""
        return float(np.interp(t, self._xs, self._ys))

    def resample(self, ts: "Timeseries") -> "Timeseries":
        """Resample to another timeseries."""
        return Timeseries(self._id, ts.xs, [self.value(t) for t in ts.xs])

    def __call__(self, t: float) -> float:
        return self.value(t)

    def __len__(self) -> int:
        return len(self._xs)


# ---------------------------------------------------
# region Curve
# ---------------------------------------------------


class Curve:
    """Static x-y curve (e.g. storage curves), linear interpolation."""

    def __init__(self, cid: str, xs: ArrayLike, ys: ArrayLike):
        self._id = cid
        self._xs = np.asarray(xs, dtype=float).flatten()
        self._ys = np.asarray(ys, dtype=float).flatten()

    @property
    def id(self) -> str:
        return self._id

    def value(self, x: float) -> float:
        return float(np.interp(x, self._xs, self._ys))

    def resample(self, xs: ArrayLike) -> "Curve":
        return Curve(self._id, xs, [self.value(x) for x in xs])

    def __call__(self, x: float) -> float:
        return self.value(x)

    def __len__(self) -> int:
        return len(self._xs)


# ---------------------------------------------------
# region Pattern
# ---------------------------------------------------


class Pattern:
    """
    A pattern is a reusable, periodic timeseries.
    """

    def __init__(self, id: str): ...

    @property
    def id(self) -> str: ...

    def value(self, t: float) -> float: ...

    def resample(self, mode: str) -> "Pattern": ...

    def __call__(self, t: float) -> float:
        return self.value(t)

    def __len__(self) -> int: ...


# ---------------------------------------------------
# region Table
# ---------------------------------------------------


class Table:
    """
    A multidimensional lookup table supporting numerical and categorical axes.
    """

    def __init__(self, id: str): ...

    @property
    def id(self) -> str: ...

    def value(self, x: Hashable, y: Hashable, z: Hashable = None) -> Any: ...

    def resample(self, x: Hashable, y: Hashable, z: Hashable = None) -> "Table": ...

    def __call__(self, x: Hashable, y: Hashable, z: Hashable = None) -> Any:
        return self.value(x, y, z)

    def __len__(self) -> int: ...
