# -*- encoding: utf-8 -*-
"""
Copyright (C) 2025, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Time-, curve- and pattern-series data structures.

NOTE: Do not use one series instance in multiple places to avoid confusion.
"""
import numpy as np
from typing import Hashable, Any
from dateutil.parser import parse


class Timeseries:
    """
    A timeseries is a sequence of data points indexed by time.
    """

    def __init__(self, id: str, time: np.ndarray, data: np.ndarray):
        """Timeseries.

        Args:
            id: The id of the timeseries.
            time: The time array, must be monotonic, increasing and in seconds.
            data: The data array, must be 1d array.
        """
        self._data = data.flatten()
        self._time = time.flatten()
        self._id = id

        self._check()

    def _check(self):
        """Check time and data validity."""
        if len(self._time) != len(self._data):
            raise ValueError("Time and data length mismatch.")

        if not np.all(np.diff(self._time) > 0):
            raise ValueError("Time is not monotonic.")

    def __len__(self):
        return len(self._time)

    @staticmethod
    def from_expr(
        id: str, func: str, start: str, end: str, step: float
    ) -> "Timeseries":
        """Create a timeseries from an expression."""
        t0 = parse(start).timestamp()
        t1 = parse(end).timestamp()
        ts = np.arange(t0, t1 + step, step)
        ys = np.array([eval(func)(t) for t in ts])
        return Timeseries(id, ts, ys)

    @property
    def id(self) -> str:
        """Return the id of the timeseries."""
        return self._id

    @property
    def time_range(self) -> tuple[float, float]:
        """Return the time range."""
        return (self._time[0], self._time[-1])

    @property
    def data_range(self) -> tuple[float, float]:
        """Return the data range."""
        return (np.min(self._data), np.max(self._data))

    def get_value(self, time: float, interp: str = "linear") -> float:
        """Get the data value at a specific time.

        Args:
            time: The time to get the data value.
            interp: The interpolation method, ["linear", "nearest].
        """
        if time < self._time[0] or time > self._time[-1]:
            raise ValueError("Time out of range.")

        if interp == "nearest":
            idx = np.abs(self._time - time).argmin()
            return self._data[idx]
        return np.interp(time, self._time, self._data)

    def resample(self, step: float) -> "Timeseries":
        """Resample with uniform time step."""
        new_time = np.arange(self._time[0], self._time[-1], step)
        new_data = np.interp(new_time, self._time, self._data)
        return Timeseries(new_time, new_data)


class Curve:
    """
    A curve is a function that maps two float dataseries.
    """

    def __init__(self, id: str, xs: np.ndarray, ys: np.ndarray):
        """Curve.

        Args:
            id: The id of the curve.
            xs: The x array, must be monotonic.
            ys: The y array, must be 1d array.
        """
        self._xs = xs.flatten()
        self._ys = ys.flatten()
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

    @staticmethod
    def from_expr(expr: str) -> "Curve":
        """Create a curve from an expression.

        NOTE: To be implemented later.
        """
        pass

    @property
    def id(self) -> str:
        """Return the id of the curve."""
        return self._id

    @property
    def xs_range(self) -> tuple[float, float]:
        """Return the xs range."""
        return (np.min(self._xs), np.max(self._xs))

    @property
    def ys_range(self) -> tuple[float, float]:
        """Return the ys range."""
        return (np.min(self._ys), np.max(self._ys))

    def get_value(self, x: float) -> float:
        """Get the y value at a specific x."""
        if x < self._xs[0] or x > self._xs[-1]:
            raise ValueError("X out of range.")

        return np.interp(x, self._xs, self._ys)

    def inverse(self, y: float) -> float:
        """Get the x value at a specific y."""
        if not self._is_one2one:
            raise ValueError("Curve is not one-to-one, cannot inverse.")

        if y < self._ys[0] or y > self._ys[-1]:
            raise ValueError("Y out of range.")

        return np.interp(y, self._ys, self._xs)

    def resample(self, step: float) -> "Curve":
        """Resample with uniform x step."""
        new_xs = np.arange(self._xs[0], self._xs[-1], step)
        new_ys = np.interp(new_xs, self._xs, self._ys)
        return Curve(self._id, new_xs, new_ys)


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
        self._data = data.flatten()

        self._modes = {"s": 1, "h": 3600, "d": 86400, "m": 2592000}
        dt = self._modes[mode]
        self._time = np.array(
            [i * dt for i in range(len(self._data))],
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
        """Return the period."""
        return self._time[-1]

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

        self._x = x.flatten()
        self._y = y.flatten() if y is not None else None
        self._z = z.flatten() if z is not None else None
        self._data = data

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
        index = [None, None, None]

        xi = np.where(self._x == x)[0]
        if len(xi) == 0:
            raise ValueError("X value not found in table.")
        index[0] = xi[0]

        if self.ndim > 1:
            yi = np.where(self._y == y)[0]
            if len(yi) == 0:
                raise ValueError("Y value not found in table.")
            index[1] = yi[0]

        if self.ndim > 2:
            zi = np.where(self._z == z)[0]
            if len(zi) == 0:
                raise ValueError("Z value not found in table.")
            index[2] = zi[0]

        return self._data[tuple(index)]
