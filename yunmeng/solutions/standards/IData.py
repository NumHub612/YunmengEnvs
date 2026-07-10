# -*- encoding: utf-8 -*-
"""
Copyright (C) 2025, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Data-description interfaces for exchange items.

This module provides the "Where, When, What" descriptors attached to
every piece of data flowing between components:
  - ITimeSpan     : temporal domain and resolution
  - IElementSet   : spatial domain (elements, coordinates)
  - IQuantity     : physical quantity (name, unit, dtype)
  - IValueSet     : the actual data container
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional
import numpy as np

# ---------------------------------------------------
# region ITimeSpan
# ---------------------------------------------------


@dataclass
class ITimeSpan:
    """Time span attached to an exchange item.

    For a scalar time-series output this describes the full horizon;
    for a field that evolves in time the step size drives
    the coupling frequency between components.
    """

    start: float = 0.0
    """Start timestamp (seconds since epoch, or datetime timestamp)."""

    end: Optional[float] = None
    """End timestamp; None means open-ended."""

    step: float = 3600.0
    """Intended time-step size in seconds."""

    timezone_offset: float = 0.0
    """Offset from UTC in hours."""

    @property
    def duration(self) -> Optional[float]:
        """Total duration in seconds."""
        if self.end is None:
            return None
        return self.end - self.start

    def steps_count(self) -> Optional[int]:
        """Number of steps."""
        if self.end is None:
            return None
        return int((self.end - self.start) / self.step)

    def timestamp(self, step_idx: int) -> float:
        """The wall-clock time of *step_idx*."""
        return self.start + step_idx * self.step


# ---------------------------------------------------
# region GeoReference
# ---------------------------------------------------


@dataclass
class GeoReference:
    """Lightweight geographic coordinate reference."""

    crs: str = ""
    """Coordinate reference system as OGC WKT."""

    origin_x: float = 0.0
    origin_y: float = 0.0
    origin_z: float = 0.0


# ---------------------------------------------------
# region IElementSet
# ---------------------------------------------------


class IElementSet(ABC):
    """Spatial element set describing WHERE data lives.

    Supports three usage patterns:
      1. Scalar   : element_count == 1, no spatial info(bulk value).
      2. Points   : ordered list of discrete point locations.
      3. Mesh     : structured or unstructured grid.
    """

    @property
    @abstractmethod
    def element_count(self) -> int:
        """Number of spatial elements."""
        pass

    @property
    @abstractmethod
    def element_type(self) -> str:
        """Element geometry type.

        One of: scalar, point, polyline, polygon, polyhedron.
        """
        pass

    @abstractmethod
    def get_coordinates(self, element_index: int) -> np.ndarray:
        """Return coordinates of element vertices."""
        pass

    @abstractmethod
    def get_center(self, element_index: int) -> np.ndarray:
        """Return element centroid, shape (3,)."""
        pass

    @property
    def geo_reference(self) -> GeoReference:
        """Geographic CRS info."""
        return GeoReference()


# ---------------------------------------------------
# region IQuantity
# ---------------------------------------------------


@dataclass
class IQuantity:
    """Physical quantity metadata: name, unit, data type."""

    name: str = ""
    """Variable name, e.g. "discharge", "water_depth"."""

    description: str = ""

    unit: str = ""
    """Physical unit as a SI string, e.g. "m3/s", "m".
    An empty string means dimensionless."""

    dtype: str = "float64"
    """Numpy-style dtype string."""

    missing_value: Any = np.nan
    """Value used to represent missing / invalid data."""

    def convert_factor_to(self, target_unit: str) -> float:
        """Return the multiplicative factor to convert
        from *self.unit* to *target_unit*.
        """
        if self.unit == target_unit:
            return 1.0
        raise NotImplementedError()


# ---------------------------------------------------
# region IValueSet
# ---------------------------------------------------


class IValueSet(ABC):
    """Value set storing the actual numeric data of an exchange item.

    Conceptual shape: (time_steps, elements, components).
    For a scalar time series: shape (T, 1, 1).
    For a 2-D field with one variable: shape (T, N_cells, 1).
    """

    @property
    @abstractmethod
    def quantity(self) -> IQuantity:
        """Physical quantity definition (What)."""
        pass

    @property
    @abstractmethod
    def shape(self) -> tuple:
        """Current shape as (time_steps, elements, components)."""
        pass

    @abstractmethod
    def get_values(self, time_idx: int = -1) -> np.ndarray:
        """Retrieve values at the given time index."""
        pass

    @abstractmethod
    def set_values(self, time_idx: int, values: np.ndarray):
        """Overwrite values at a specific time index."""
        pass

    @abstractmethod
    def append_values(self, values: np.ndarray):
        """Append a new time slice. *values* must have shape
        (elements, components)."""
        pass
