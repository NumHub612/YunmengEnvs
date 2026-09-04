# -*- encoding: utf-8 -*-
"""
Copyright (C) 2026, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Data-description interfaces for exchange items ("Where, When, What").
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from yunmeng.interfaces.types import ArrayLike, GeometryType

# ---------------------------------------------------
# region TimeSpan
# ---------------------------------------------------


@dataclass(frozen=True)
class TimeSpan:
    """Time span attached to an exchange item."""

    start: float = 0.0
    end: float | None = None  # None = open-ended
    step: float = 3600.0
    timezone_offset: float = 0.0

    @property
    def duration(self) -> float:
        """Total duration in seconds."""
        if self.end is None:
            return None
        return self.end - self.start

    def steps_count(self) -> int:
        """Number of steps."""
        if self.end is None:
            return None
        return int((self.end - self.start) / self.step)

    def timestamp(self, step_idx: int) -> float:
        """The wall-clock time of *step_idx*."""
        return self.start + step_idx * self.step


# ---------------------------------------------------
# region IElementSet
# ---------------------------------------------------


class IElementSet(ABC):
    """Spatial element set describing WHERE data lives.

    Supports three usage patterns:
      1. Id-based : element_count == 1, no spatial info(bulk value).
      2. Points   : ordered list of discrete point locations.
      3. Mesh     : structured or unstructured grid.
    """

    @property
    @abstractmethod
    def element_count(self) -> int: ...

    @property
    @abstractmethod
    def gtype(self) -> GeometryType: ...

    @abstractmethod
    def get_center(self, element_index: int) -> ArrayLike:
        """Element centroid, shape (3,)."""
        ...

    @abstractmethod
    def get_coordinates(self, element_index: int) -> ArrayLike:
        """Coordinates of element vertices."""
        ...


# ---------------------------------------------------
# region Quantity
# ---------------------------------------------------


@dataclass(frozen=True)
class Quantity:
    """Physical quantity metadata."""

    name: str = ""
    """Variable name, e.g. "discharge", "water_depth"."""

    description: str = ""

    unit: str = ""
    """Physical unit as a SI string, e.g. "m3/s", "m".
    An empty string means dimensionless."""

    dtype: str = "float64"
    """Numpy-style dtype string."""

    missing_value: Any = float("nan")
    """Value used to represent missing / invalid data."""

    si_factor: float = 1.0
    """Multiplicative factor to convert to SI unit."""

    si_offset: float = 0.0
    """Additive factor to convert to SI unit."""

    def to_si(self, value: float) -> float:
        """Convert value to SI unit."""
        return self.si_factor * value + self.si_offset


# ---------------------------------------------------
# region IValueSet
# ---------------------------------------------------


class IValueSet(ABC):
    """Value container of an exchange item."""

    @property
    @abstractmethod
    def quantity(self) -> Quantity: ...

    @property
    @abstractmethod
    def shape(self) -> tuple:
        """Get shape as(time_steps, elements, components)."""
        ...

    @abstractmethod
    def get_values(self, time_idx: int = -1) -> ArrayLike:
        """Retrieve values at the given time index."""
        ...

    @abstractmethod
    def set_values(self, time_idx: int, values: ArrayLike):
        """Overwrite values at a specific time index."""
        ...

    @abstractmethod
    def append_values(self, values: ArrayLike):
        """Append a new time slice. values must have shape
        (elements, components)."""
        ...
