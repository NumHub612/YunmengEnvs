# -*- encoding: utf-8 -*-
"""
Copyright (C) 2025, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Time-, curve- and pattern-series data structures.
"""

import numpy as np
from typing import Hashable, Any
from dateutil.parser import parse
import math
import numpy as np
from typing import Hashable, Any

# ---------------------------------------------------
# region Mathmetics
# ---------------------------------------------------

_ALLOWED_FUNCS = {
    name: getattr(math, name)
    for name in (
        "sin",
        "cos",
        "tan",
        "exp",
        "log",
        "sqrt",
        "fabs",
        "floor",
        "ceil",
        "pow",
    )
}
_ALLOWED_FUNCS.update({"abs": abs, "min": min, "max": max, "pi": math.pi, "e": math.e})


def _safe_callable(expr: str):
    code = compile(expr, "<expr>", "eval")

    def f(t: float) -> float:
        env = dict(_ALLOWED_FUNCS)
        env["t"] = t
        return float(eval(code, {"__builtins__": {}}, env))

    return f


# ---------------------------------------------------
# region Timeseries
# ---------------------------------------------------


class Timeseries:
    """A sequence of data points indexed by (monotonic increasing) time
    in seconds."""

    def __init__(self, id: str, time: np.ndarray, data: np.ndarray):
        self._data = np.asarray(data, dtype=float).flatten()
        self._time = np.asarray(time, dtype=float).flatten()
        self._id = id
        self._check()

    def _check(self):
        if len(self._time) != len(self._data):
            raise ValueError("Time and data length mismatch.")
        if len(self._time) == 0:
            raise ValueError("Empty timeseries.")
        if len(self._time) > 1 and not np.all(np.diff(self._time) > 0):
            raise ValueError("Time is not monotonic increasing.")

    @staticmethod
    def from_expr(
        id: str, func: str, start: float, end: float, step: float
    ) -> "Timeseries":
        """Create a timeseries from an expression in variable ``t``.

        ``start``/``end`` are timestamps in seconds.  The expression is
        evaluated in a restricted math namespace (no builtins).
        """
        f = _safe_callable(func)
        ts = np.arange(start, end + 0.5 * step, step)
        ys = np.array([f(t) for t in ts])
        return Timeseries(id, ts, ys)

    def __len__(self):
        return len(self._time)

    @property
    def id(self) -> str:
        return self._id

    @property
    def time(self) -> np.ndarray:
        return self._time.copy()

    @property
    def data(self) -> np.ndarray:
        return self._data.copy()

    @property
    def time_range(self) -> tuple[float, float]:
        return (self._time[0], self._time[-1])

    @property
    def data_range(self) -> tuple[float, float]:
        return (float(np.min(self._data)), float(np.max(self._data)))

    def value_at(self, index: int) -> float:
        """Value at integer position (used by step-driven models)."""
        return float(self._data[index])

    def get_value(self, time: float, interp: str = "linear") -> float:
        """Interpolated value at an arbitrary timestamp."""
        if interp == "nearest":
            idx = int(np.abs(self._time - time).argmin())
            return float(self._data[idx])
        return float(np.interp(time, self._time, self._data))

    def resample(self, id: str, step: float) -> "Timeseries":
        new_time = np.arange(self._time[0], self._time[-1], step)
        new_data = np.interp(new_time, self._time, self._data)
        return Timeseries(id, new_time, new_data)


# ---------------------------------------------------
# region Curve
# ---------------------------------------------------


class Curve:
    """A monotonic 1-D lookup function y = f(x)."""

    def __init__(self, id: str, xs: np.ndarray, ys: np.ndarray):
        self._xs = np.asarray(xs, dtype=float).flatten()
        self._ys = np.asarray(ys, dtype=float).flatten()
        self._id = id
        self._is_one2one = False
        self._check()

    def _check(self):
        if len(self._xs) != len(self._ys):
            raise ValueError("Curve xs and ys length mismatch.")
        if not np.all(np.diff(self._xs) > 0) and not np.all(np.diff(self._xs) < 0):
            raise ValueError("Curve xs is not monotonic.")
        if np.all(np.diff(self._ys) > 0) or np.all(np.diff(self._ys) < 0):
            self._is_one2one = True

    def __len__(self):
        return len(self._xs)

    @property
    def id(self) -> str:
        return self._id

    @property
    def xs_range(self) -> tuple[float, float]:
        return (float(np.min(self._xs)), float(np.max(self._xs)))

    @property
    def ys_range(self) -> tuple[float, float]:
        return (float(np.min(self._ys)), float(np.max(self._ys)))

    def get_value(self, x: float) -> float:
        return float(np.interp(x, self._xs, self._ys))

    def inverse(self, y: float) -> float:
        if not self._is_one2one:
            raise ValueError("Curve is not one-to-one, cannot invert.")
        return float(np.interp(y, self._ys, self._xs))

    def resample(self, step: float) -> "Curve":
        new_xs = np.arange(self._xs[0], self._xs[-1], step)
        new_ys = np.interp(new_xs, self._xs, self._ys)
        return Curve(self._id, new_xs, new_ys)


# ---------------------------------------------------
# region Pattern
# ---------------------------------------------------


class Pattern:
    """
    A pattern is a reusable, periodic timeseries.
    """

    def __init__(self, id: str, mode: str, data: np.ndarray):
        """Pattern.

        Args:
            id: The id of the pattern.
            mode: The pattern mode, one of ["s", "h", "d", "m"].
            data: The data array, must be 1d array.
        """
        assert mode in ["s", "h", "d", "m"], "Invalid pattern mode."
        self._id = id
        self._data = np.asarray(data).flatten()

        self._modes = {"s": 1, "h": 3600, "d": 86400, "m": 2592000}
        self._dt = self._modes[mode]
        self._time = np.array(
            [i * self._dt for i in range(len(self._data))],
        )

    def __len__(self):
        return len(self._data)

    @staticmethod
    def from_expr(expr: str, mode: str) -> "Pattern":
        """Create a pattern from an expression.

        NOTE: To be implemented later.
        """
        pass

    @property
    def id(self) -> str:
        """Return the id of the pattern."""
        return self._id

    @property
    def period(self) -> float:
        return len(self._data) * self._dt

    def get_value(self, t: float) -> float:
        """Get pattern value at relative time t by nearest interp."""
        phase = t % self.period
        idx = np.abs(self._time - phase).argmin()
        return self._data[idx]

    def resample(self, mode: str) -> "Pattern":
        """Resample the pattern to new time step."""
        if mode not in self._modes:
            raise ValueError(f"Invalid pattern mode: {mode}.")

        step = self._modes[mode]
        new_time = np.arange(0, self.period, step)
        new_data = np.interp(new_time, self._time, self._data)
        return Pattern(self._id, mode, new_data)


# ---------------------------------------------------
# region Table
# ---------------------------------------------------


class Table:
    """
    A multidimensional lookup table supporting numerical and categorical axes.
    """

    def __init__(
        self,
        id: str,
        data: np.ndarray,
        x: np.ndarray,
        y: np.ndarray = None,
        z: np.ndarray = None,
    ):
        self._id = id

        self._x = np.asarray(x).flatten()
        self._y = np.asarray(y).flatten() if y is not None else None
        self._z = np.asarray(z).flatten() if z is not None else None
        self._data = np.asarray(data)

        self._check()

    def _check(self):
        if self._x.shape[0] != self._data.shape[0]:
            raise ValueError("Table x dimension mismatch.")

        ndim = 1
        if self._y is not None:
            ndim += 1
        if self._z is not None:
            ndim += 1
        if self._data.ndim != ndim:
            raise ValueError("Table data dimension mismatch.")

        if ndim > 1 and self._y.shape[0] != self._data.shape[1]:
            raise ValueError("Table y dimension mismatch.")
        if ndim > 2 and self._z.shape[0] != self._data.shape[2]:
            raise ValueError("Table z dimension mismatch.")

    @staticmethod
    def from_expr(expr: str) -> "Table":
        """Create a table from an expression.

        NOTE: To be implemented later.
        """
        pass

    @property
    def id(self) -> str:
        return self._id

    @property
    def ndim(self) -> int:
        return self._data.ndim

    @property
    def shape(self) -> tuple[int, int, int]:
        return self._data.shape

    def get_value(
        self,
        x: Hashable,
        y: Hashable = None,
        z: Hashable = None,
    ) -> Any:
        """Get the table value at specific coordinates."""
        index = []

        xi = np.where(self._x == x)[0]
        if len(xi) == 0:
            raise ValueError("X value not found in table.")
        index.append(xi[0])

        if self.ndim > 1:
            yi = np.where(self._y == y)[0]
            if len(yi) == 0:
                raise ValueError("Y value not found in table.")
            index.append(yi[0])

        if self.ndim > 2:
            zi = np.where(self._z == z)[0]
            if len(zi) == 0:
                raise ValueError("Z value not found in table.")
            index.append(zi[0])

        return self._data[tuple(index)]
